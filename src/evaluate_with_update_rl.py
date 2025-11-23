#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 21:11:38 2025

@author: ivanovsi
"""
import os
import torch
import pandas as pd
import numpy as np
from lstm_q_agent import LSTMQAgent
from rbc_agent import RBCAgent
from agent_setup import load_all_building_data

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------
DATA_ROOT = '../outputs/phase3'
OUTPUT_DIR = '../outputs/eval/comprehensive_metrics'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 17
HIDDEN_DIM = 64
OUTPUT_DIM = 5
SEQ_LEN = 8
TEST_BUILDINGS = ['Building_5', 'Building_6']

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print("="*80)
print("METRICS EVALUATION: Meta-RL vs Standard RL vs RBC")
print("="*80)


# ---------------------------------------------------------------------
# ACTION MAPPING
# ---------------------------------------------------------------------
def map_action_to_continuous(action_idx):
    """Map discrete action to continuous range [-1.0, 1.0]"""
    return -1.0 + 2.0 * action_idx / (OUTPUT_DIM - 1)


# ---------------------------------------------------------------------
# METRIC CALCULATION
# ---------------------------------------------------------------------
def calculate_comprehensive_metrics(df, actions_continuous, actions_discrete, 
                                    label, building_name):
    """
    Calculate comprehensive performance metrics:
    - Carbon, Cost, Peak reduction
    - Ramping (smoothness)
    - Load Factor
    - Average Peak Demand
    - Annual Peak Demand
    - Net electricity consumption
    """

    # === BASELINE (no control) ===
    baseline_cooling = df['cooling_demand'].values
    baseline_heating = df['heating_demand'].values
    baseline_non_shiftable = df['non_shiftable_load'].values

    baseline_consumption = baseline_cooling + baseline_heating + baseline_non_shiftable
    baseline_carbon = (df['carbon_intensity'] * baseline_consumption).sum()
    baseline_cost = (df['electricity_pricing'] * baseline_consumption).sum()
    baseline_peak = baseline_consumption.max()

    # === CONTROLLED (with actions) ===
    action_series = pd.Series(actions_continuous + [0.0] * SEQ_LEN)[:len(df)]

    controlled_cooling = df['cooling_demand'] * (1 + 0.1 * action_series)
    controlled_heating = df['heating_demand'] * (1 - 0.1 * action_series)
    controlled_consumption = (controlled_cooling + controlled_heating +
                              df['non_shiftable_load']).values

    controlled_carbon = (df['carbon_intensity'] * controlled_consumption).sum()
    controlled_cost = (df['electricity_pricing'] * controlled_consumption).sum()
    controlled_peak = controlled_consumption.max()

    # === Metric 1–3: Carbon, Cost, Peak ===
    carbon_reduction = ((baseline_carbon - controlled_carbon) / baseline_carbon) * 100
    cost_savings = ((baseline_cost - controlled_cost) / baseline_cost) * 100
    peak_reduction = ((baseline_peak - controlled_peak) / baseline_peak) * 100

    # === Metric 4: Ramping (action smoothness) ===
    if len(actions_discrete) > 1:
        action_diffs = np.abs(np.diff(actions_discrete))
        ramping = np.mean(action_diffs)
    else:
        ramping = 0.0

    ramping_norm = ramping / 4.0  # max difference = 4

    # === Metric 5: Load Factor ===
    avg_load = np.mean(controlled_consumption)
    peak_load = np.max(controlled_consumption)
    load_factor = avg_load / peak_load if peak_load > 0 else 0

    baseline_lf = np.mean(baseline_consumption) / np.max(baseline_consumption)
    load_factor_norm = (1 - load_factor) / (1 - baseline_lf)

    # === Metric 6: Average Daily Peak ===
    hours_per_day = 24
    num_days = len(controlled_consumption) // hours_per_day

    daily_peaks = [
        np.max(controlled_consumption[d*24:(d+1)*24])
        for d in range(num_days)
    ]
    baseline_daily_peaks = [
        np.max(baseline_consumption[d*24:(d+1)*24])
        for d in range(num_days)
    ]

    avg_peak_norm = np.mean(daily_peaks) / np.mean(baseline_daily_peaks)

    # === Metric 7: Annual Peak ===
    annual_peak_norm = controlled_peak / baseline_peak

    # === Metric 8: Net Consumption ===
    net_consumption_norm = np.sum(controlled_consumption) / np.sum(baseline_consumption)

    return {
        'building': building_name,
        'label': label,
        'carbon_reduction_pct': carbon_reduction,
        'cost_savings_pct': cost_savings,
        'peak_reduction_pct': peak_reduction,
        'ramping_norm': ramping_norm,
        'load_factor_norm': load_factor_norm,
        'avg_peak_norm': avg_peak_norm,
        'annual_peak_norm': annual_peak_norm,
        'net_consumption_norm': net_consumption_norm,
    }


# ---------------------------------------------------------------------
# LSTM-Q AGENT EVALUATION (Meta-RL + Standard RL)
# ---------------------------------------------------------------------
def evaluate_lstm_agent(agent_path, label):
    print(f"\nEvaluating {label}...")

    agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    agent.model.load_state_dict(torch.load(agent_path))
    agent.model.eval()

    all_tasks = load_all_building_data(DATA_ROOT)
    test_tasks = all_tasks[4:6]

    results = []

    for i, df in enumerate(test_tasks):
        building_name = TEST_BUILDINGS[i]

        # Create sequences
        sequences = [
            torch.tensor(df.iloc[j:j+SEQ_LEN].values, dtype=torch.float32).unsqueeze(0)
            for j in range(len(df) - SEQ_LEN)
        ]

        actions_discrete = []
        actions_continuous = []

        for seq in sequences:
            q_vals = agent.predict(seq).squeeze().detach().numpy()
            a = np.argmax(q_vals)
            a_cont = map_action_to_continuous(a)

            actions_discrete.append(a)
            actions_continuous.append(a_cont)

        metrics = calculate_comprehensive_metrics(
            df, actions_continuous, actions_discrete, label, building_name
        )
        results.append(metrics)

    return results


# ---------------------------------------------------------------------
# RBC BASELINE
# ---------------------------------------------------------------------
def evaluate_rbc_agent():
    print("\nEvaluating RBC...")

    agent = RBCAgent()
    all_tasks = load_all_building_data(DATA_ROOT)
    test_tasks = all_tasks[4:6]

    results = []

    for i, df in enumerate(test_tasks):
        building_name = TEST_BUILDINGS[i]

        actions_discrete = []
        actions_continuous = []

        for j in range(len(df) - SEQ_LEN):
            state = df.iloc[j + SEQ_LEN].values
            a = agent.predict(state)
            a_cont = map_action_to_continuous(a)

            actions_discrete.append(a)
            actions_continuous.append(a_cont)

        metrics = calculate_comprehensive_metrics(
            df, actions_continuous, actions_discrete, 'RBC', building_name
        )
        results.append(metrics)

    return results


# ---------------------------------------------------------------------
# RUN EVALUATION
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("EVALUATING ALL METHODS")
print("="*80)

meta_results = evaluate_lstm_agent('../models/meta_rl_model.pt', 'Meta-RL')
rl_results = evaluate_lstm_agent('../models/rl_lstm_q_model.pt', 'Standard RL')
rbc_results = evaluate_rbc_agent()

# ---------------------------------------------------------------------
# AVERAGE RESULTS
# ---------------------------------------------------------------------
print("\n" + "="*80)
print("AVERAGED RESULTS")
print("="*80)

methods = {
    'Meta-RL': meta_results,
    'Standard RL': rl_results,
    'RBC': rbc_results
}

summary = []

for method_name, results in methods.items():
    avg = {
        'Method': method_name,
        'Carbon_%': np.mean([r['carbon_reduction_pct'] for r in results]),
        'Cost_%': np.mean([r['cost_savings_pct'] for r in results]),
        'Peak_%': np.mean([r['peak_reduction_pct'] for r in results]),
        'Ramping': np.mean([r['ramping_norm'] for r in results]),
        'Load_Factor': np.mean([r['load_factor_norm'] for r in results]),
        'Avg_Peak': np.mean([r['avg_peak_norm'] for r in results]),
        'Annual_Peak': np.mean([r['annual_peak_norm'] for r in results]),
        'Net_Consumption': np.mean([r['net_consumption_norm'] for r in results]),
    }
    summary.append(avg)

summary_df = pd.DataFrame(summary)
summary_df.to_csv(f'{OUTPUT_DIR}/all_metrics.csv', index=False)

print(f"\nSaved: {OUTPUT_DIR}/all_metrics.csv")


print("\n" + "="*80)
print("EVALUATING ALL METHODS (Meta-RL, Standard RL, RBC)")
print("This step loads each agent and computes performance metrics on buildings 5–6.")
print("="*80)

# -----------------------------
# Evaluate Meta-RL agent
# -----------------------------
print("\n>>> Evaluating Meta-RL LSTM-Q Agent")
meta_results = evaluate_lstm_agent('../models/meta_rl_model.pt', 'Meta-RL')

# -----------------------------
# Evaluate Standard RL agent
# -----------------------------
print("\n>>> Evaluating Standard RL LSTM-Q Agent")
rl_results = evaluate_lstm_agent('../models/rl_lstm_q_model.pt', 'Standard RL')

# -----------------------------
# Evaluate RBC agent
# -----------------------------
print("\n>>> Evaluating Rule-Based Control (RBC) Agent")
rbc_results = evaluate_rbc_agent()

print("\n" + "="*80)
print("AVERAGED RESULTS ACROSS TEST BUILDINGS (5–6)")
print("="*80)

methods = {
    'Meta-RL': meta_results,
    'Standard RL': rl_results,
    'RBC': rbc_results
}

summary = []

for method_name, results in methods.items():
    print(f"\n>>> Aggregating metrics for {method_name}")
    
    avg_metrics = {
        'Method': method_name,
        'Carbon_%': np.mean([r['carbon_reduction_pct'] for r in results]),
        'Cost_%': np.mean([r['cost_savings_pct'] for r in results]),
        'Peak_%': np.mean([r['peak_reduction_pct'] for r in results]),
        'Ramping': np.mean([r['ramping_norm'] for r in results]),
        'Load_Factor': np.mean([r['load_factor_norm'] for r in results]),
        'Avg_Peak': np.mean([r['avg_peak_norm'] for r in results]),
        'Annual_Peak': np.mean([r['annual_peak_norm'] for r in results]),
        'Net_Consumption': np.mean([r['net_consumption_norm'] for r in results]),
    }
    summary.append(avg_metrics)

    # Print summary for each method
    print(f"  Carbon Reduction:      {avg_metrics['Carbon_%']:.2f}%")
    print(f"  Cost Savings:          {avg_metrics['Cost_%']:.2f}%")
    print(f"  Peak Reduction:        {avg_metrics['Peak_%']:.2f}%")
    print(f"  Ramping (Normalized):  {avg_metrics['Ramping']:.3f}")
    print(f"  Load Factor (Norm.):   {avg_metrics['Load_Factor']:.3f}")
    print(f"  Avg Peak (Norm.):      {avg_metrics['Avg_Peak']:.3f}")
    print(f"  Annual Peak (Norm.):   {avg_metrics['Annual_Peak']:.3f}")
    print(f"  Net Consumption (Norm):{avg_metrics['Net_Consumption']:.3f}")

# Save results
summary_df = pd.DataFrame(summary)
summary_path = f'{OUTPUT_DIR}/all_metrics.csv'
summary_df.to_csv(summary_path, index=False)

print(f"\n✔ Saved aggregated metrics to {summary_path}")

# -----------------------------
# Meta-RL improvement over RBC
# -----------------------------
print("\n" + "="*80)
print("META-RL IMPROVEMENT OVER RBC (Normalized Metrics)")
print("="*80)

meta_row = summary_df[summary_df['Method'] == 'Meta-RL'].iloc[0]
rbc_row = summary_df[summary_df['Method'] == 'RBC'].iloc[0]

improvements = {
    'Ramping': ((rbc_row['Ramping'] - meta_row['Ramping']) / max(rbc_row['Ramping'], 1e-6)) * 100,
    'Load_Factor': ((rbc_row['Load_Factor'] - meta_row['Load_Factor']) / max(rbc_row['Load_Factor'], 1e-6)) * 100,
    'Avg_Peak': ((rbc_row['Avg_Peak'] - meta_row['Avg_Peak']) / max(rbc_row['Avg_Peak'], 1e-6)) * 100,
    'Annual_Peak': ((rbc_row['Annual_Peak'] - meta_row['Annual_Peak']) / max(rbc_row['Annual_Peak'], 1e-6)) * 100,
    'Net_Consumption': ((rbc_row['Net_Consumption'] - meta_row['Net_Consumption']) / max(rbc_row['Net_Consumption'], 1e-6)) * 100,
}

for metric, improvement in improvements.items():
    print(f"  {metric}: {improvement:.2f}% better than RBC")

print("\n" + "="*80)
print("✔ ALL METRICS COMPUTED AND COMPARED SUCCESSFULLY")
print("="*80)
