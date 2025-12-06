#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import torch
import pandas as pd
import numpy as np

from lstm_q_agent import LSTMQAgent         # Reptile Meta-RL model
from sac_agent import SACAgent              # Standard RL model
from rbc_agent import RBCAgent              # Rule-based baseline

from agent_setup import load_all_building_data
from environment_wrapper import SimpleBuildingEnv
from recurrent_env_wrapper import RecurrentBuildingEnv

DATA_ROOT   = '../outputs/phase3'
OUTPUT_DIR  = '../outputs/comprehensive_metrics'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Meta-RL (Recurrent LSTM-Q)
INPUT_DIM_LSTM  = 23        # state + prev_action + prev_reward
HIDDEN_DIM_LSTM = 128
OUTPUT_DIM      = 5
SEQ_LEN         = 8

# SAC
INPUT_DIM_SAC   = 17 * 8     # flattened window
HIDDEN_DIM_SAC  = 512

TEST_BUILDINGS  = ['Building_5', 'Building_6']

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)



def map_action_to_continuous(action_idx: int):
    """Discrete index 0–4 -> continuous action [-1, 1]"""
    return -1.0 + 2.0 * action_idx / (OUTPUT_DIM - 1)


def calculate_comprehensive_metrics(df, actions_cont, actions_disc, label, b_name):
    """Computes all consumption-based and temporal metrics."""

    # --- Align actions with timeseries length
    pad_len = len(df) - len(actions_cont)
    if pad_len > 0:
        actions_cont += [0.0] * pad_len
    
    action_series = pd.Series(actions_cont, index=df.index)

    # === BASELINE (no control) ===
    c0 = df['cooling_demand'].values
    h0 = df['heating_demand'].values
    ns = df['non_shiftable_load'].values

    base_consumption = c0 + h0 + ns
    base_carbon = (df['carbon_intensity'] * base_consumption).sum()
    base_cost   = (df['electricity_pricing'] * base_consumption).sum()
    base_peak   = np.max(base_consumption)

    # === CONTROLLED ===
    c_adj = df['cooling_demand'].values * (1 + 0.1 * action_series.values)
    h_adj = df['heating_demand'].values * (1 - 0.1 * action_series.values)

    controlled = c_adj + h_adj + ns
    controlled = np.nan_to_num(controlled, nan=np.nanmean(controlled))

    ctrl_carbon = (df['carbon_intensity'] * controlled).sum()
    ctrl_cost   = (df['electricity_pricing'] * controlled).sum()
    ctrl_peak   = np.max(controlled)

    carbon_reduction = (base_carbon - ctrl_carbon) / base_carbon * 100
    cost_savings     = (base_cost - ctrl_cost) / base_cost * 100
    peak_reduction   = (base_peak - ctrl_peak) / base_peak * 100

    # === RAMPING ===
    if len(actions_disc) > 1:
        diffs = np.abs(np.diff(actions_disc))
        ramp = np.mean(diffs)
    else:
        ramp = 0.0

    ramp_norm = ramp / 4.0     # max jump 0→4

    # === LOAD FACTOR ===
    avg_load = np.mean(controlled)
    peak_load = np.max(controlled)
    lf = avg_load / peak_load

    base_lf = np.mean(base_consumption) / np.max(base_consumption)

    lf_norm = (1 - lf) / (1 - base_lf)

    # === AVERAGE DAILY PEAK ===
    hours = 24
    days = len(controlled) // hours
    daily_peaks = [np.max(controlled[d*hours:(d+1)*hours]) for d in range(days)]
    base_dpeaks = [np.max(base_consumption[d*hours:(d+1)*hours]) for d in range(days)]

    avg_peak_norm = (np.mean(daily_peaks) / np.mean(base_dpeaks))

    # === ANNUAL PEAK ===
    annual_peak_norm = ctrl_peak / base_peak

    # === NET CONSUMPTION ===
    net_norm = controlled.sum() / base_consumption.sum()

    return {
        'building': b_name,
        'label': label,
        'carbon_reduction_pct': carbon_reduction,
        'cost_savings_pct': cost_savings,
        'peak_reduction_pct': peak_reduction,
        'ramping_norm': ramp_norm,
        'load_factor_norm': lf_norm,
        'avg_peak_norm': avg_peak_norm,
        'annual_peak_norm': annual_peak_norm,
        'net_consumption_norm': net_norm,
        'ramping_raw': ramp,
        'load_factor_raw': lf,
        'avg_peak_raw': np.mean(daily_peaks),
        'annual_peak_raw': ctrl_peak,
        'net_consumption_raw': controlled.sum(),
    }


#  EVALUATE META-RL
def evaluate_metarl():
    print("\nEvaluating Meta-RL (Reptile LSTM-Q)...")

    agent = LSTMQAgent(INPUT_DIM_LSTM, HIDDEN_DIM_LSTM, OUTPUT_DIM)
    agent.model.load_state_dict(torch.load('../models/meta_lstmq_reptile.pt'))
    agent.model.eval()

    all_tasks, scaler, columns = load_all_building_data(DATA_ROOT)
    test = all_tasks[4:6]

    results = []

    for i, df in enumerate(test):
        name = TEST_BUILDINGS[i]

        env = RecurrentBuildingEnv(SimpleBuildingEnv(
            df, seq_len=SEQ_LEN, n_actions=OUTPUT_DIM,
            scaler=scaler, columns=columns
        ))

        actions_disc = []
        actions_cont = []

        s = env.reset()
        done = False
        while not done:
            st = torch.tensor(s, dtype=torch.float32).unsqueeze(0)
            q = agent.predict(st).squeeze().numpy()
            a = int(np.argmax(q))
            ac = map_action_to_continuous(a)

            actions_disc.append(a)
            actions_cont.append(ac)

            s, _, done, _ = env.step(a)

        metrics = calculate_comprehensive_metrics(df, actions_cont, actions_disc, 'Meta-RL', name)
        results.append(metrics)

        print(f"  {name}: Cost {metrics['cost_savings_pct']:.2f}% | Ramping {metrics['ramping_norm']:.3f}")

    return results


# Evaluate Standard RL (SAC)
def evaluate_sac():
    print("\nEvaluating Standard RL (SAC)...")

    from sac_agent import SACAgent
    action_space = type("A", (), {"low": np.array([-1.0]), "high": np.array([1.0]), "shape": (1,)})()
    agent = SACAgent(INPUT_DIM_SAC, action_space, hidden_dim=HIDDEN_DIM_SAC)
    agent.load_model("../models/sac_model.pt")

    all_tasks, scaler, columns = load_all_building_data(DATA_ROOT)
    test = all_tasks[4:6]

    results = []

    for i, df in enumerate(test):
        name = TEST_BUILDINGS[i]

        env = SimpleBuildingEnv(df, seq_len=SEQ_LEN, n_actions=OUTPUT_DIM,
                                scaler=scaler, columns=columns)

        actions_disc = []
        actions_cont = []

        s = env.reset().flatten()
        done = False
        while not done:
            act = agent.select_action(s, evaluate=True)[0]
            actions_cont.append(act)

            # continuous SAC action → discrete index
            disc = int((act + 1) * (OUTPUT_DIM - 1) / 2)
            disc = np.clip(disc, 0, OUTPUT_DIM - 1)
            actions_disc.append(disc)

            ns, _, done, _ = env.step(act)
            s = ns.flatten()

        metrics = calculate_comprehensive_metrics(df, actions_cont, actions_disc, "Standard RL", name)
        results.append(metrics)

        print(f"  {name}: Cost {metrics['cost_savings_pct']:.2f}% | Ramping {metrics['ramping_norm']:.3f}")

    return results


# Evaluate RBC
def evaluate_rbc():
    print("\nEvaluating RBC...")

    agent = RBCAgent()
    all_tasks, scaler, columns = load_all_building_data(DATA_ROOT)
    test = all_tasks[4:6]

    results = []

    for i, df in enumerate(test):
        name = TEST_BUILDINGS[i]

        actions_disc = []
        actions_cont = []

        for t in range(len(df) - SEQ_LEN):
            s = df.iloc[t + SEQ_LEN].values
            a = agent.predict(s)
            ac = map_action_to_continuous(a)
            actions_disc.append(a)
            actions_cont.append(ac)

        metrics = calculate_comprehensive_metrics(df, actions_cont, actions_disc, "RBC", name)
        results.append(metrics)

        print(f"  {name}: Cost {metrics['cost_savings_pct']:.2f}% | Ramping {metrics['ramping_norm']:.3f}")

    return results


# ================================================================
# RUN EVALUATION
# ================================================================
if __name__ == "__main__":
    print("\n" + "="*80)
    print("EVALUATING ALL METHODS ON REAL BUILDING DATA")
    print("="*80)

    meta = evaluate_metarl()
    sac = evaluate_sac()
    rbc = evaluate_rbc()

    # ---- Summary Table ----
    summary = []
    methods = {"Meta-RL": meta, "Standard RL": sac, "RBC": rbc}

    print("\n" + "="*80)
    print("AVERAGED RESULTS")
    print("="*80)

    for name, res in methods.items():
        avg = {
            "Method": name,
            "Carbon_%": np.mean([m['carbon_reduction_pct'] for m in res]),
            "Cost_%":   np.mean([m['cost_savings_pct'] for m in res]),
            "Peak_%":   np.mean([m['peak_reduction_pct'] for m in res]),
            "Ramping":  np.mean([m['ramping_norm'] for m in res]),
            "Load_Factor": np.mean([m['load_factor_norm'] for m in res]),
            "Avg_Peak":    np.mean([m['avg_peak_norm'] for m in res]),
            "Annual_Peak": np.mean([m['annual_peak_norm'] for m in res]),
            "Net_Consumption": np.mean([m['net_consumption_norm'] for m in res]),
        }
        summary.append(avg)

        print(f"\n{name}:")
        print(f"  Cost Savings:       {avg['Cost_%']:.2f}%")
        print(f"  Ramping (norm):     {avg['Ramping']:.3f}")
        print(f"  Load Factor (norm): {avg['Load_Factor']:.3f}")
        print(f"  Avg Peak (norm):    {avg['Avg_Peak']:.3f}")
        print(f"  Annual Peak (norm): {avg['Annual_Peak']:.3f}")
        print(f"  Net Consumption:    {avg['Net_Consumption']:.3f}")

    # Save CSV
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(f"{OUTPUT_DIR}/all_metrics.csv", index=False)
    print(f"\nSaved: {OUTPUT_DIR}/all_metrics.csv")

    # ---- Improvement over RBC ----
    print("\n" + "="*80)
    print("META-RL IMPROVEMENT OVER RBC")
    print("="*80)

    meta_row = df_summary[df_summary["Method"] == "Meta-RL"].iloc[0]
    rbc_row = df_summary[df_summary["Method"] == "RBC"].iloc[0]

    for metric in ["Ramping", "Load_Factor", "Avg_Peak", "Annual_Peak", "Net_Consumption"]:
        if rbc_row[metric] == 0:
            imp = float('inf')
        else:
            imp = (rbc_row[metric] - meta_row[metric]) / rbc_row[metric] * 100

        print(f"  {metric}: {imp:.2f}%")

    print("\n" + "="*80)
    print(" ALL METRICS CALCULATED SUCCESSFULLY ")
    print("="*80)
