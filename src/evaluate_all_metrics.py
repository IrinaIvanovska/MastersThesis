#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 21:21:16 2025

@author: ivanovsi
"""

# ----------------------------------------
# Comprehensive evaluation with ALL metrics calculated from real data:
# - Carbon, Cost, Peak 
# - Ramping (from action sequences)
# - Load Factor (from consumption timeseries)
# - Average/Annual Peak (from consumption timeseries)  
# ----------------------------------------
"""
Comprehensive Evaluation Script
Evaluates Meta-RL, Standard RL, and RBC on Test Buildings (5-6)
Generates all_metrics.csv for plotting.
"""

import os
import torch
import pandas as pd
import numpy as np
from lstm_q_agent import LSTMQAgent
from sac_agent import SACAgent
from rbc_agent import RBCAgent
from agent_setup import load_all_building_data
from environment_wrapper import SimpleBuildingEnv
from recurrent_env_wrapper import RecurrentBuildingEnv

# Config
DATA_ROOT = '../outputs/phase3'
OUTPUT_DIR = '../outputs/comprehensive_metrics'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Recurrent Meta-RL config
INPUT_DIM_LSTM = 23
HIDDEN_DIM_LSTM = 128
OUTPUT_DIM = 5
SEQ_LEN = 8

# SAC config  
INPUT_DIM_SAC = 17 * 8
HIDDEN_DIM_SAC = 512

TEST_BUILDINGS = ['Building_5', 'Building_6']
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

class ActionSpace:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

print("="*80)
print("COMPREHENSIVE METRICS EVALUATION")
print("Calculating ALL metrics from real data")
print("="*80)


def map_action_to_continuous(action_idx):
    """Map discrete action to continuous range [-1.0, 1.0]"""
    return -1.0 + 2.0 * action_idx / (OUTPUT_DIM - 1)


def calculate_comprehensive_metrics(df, actions_continuous, actions_discrete, 
                                    label, building_name):
    """
    Calculate ALL metrics from actual data:
    1. Carbon, Cost, Peak (existing)
    2. Ramping (smoothness of control)
    3. Load Factor (efficiency)
    4. Average Peak Demand
    5. Annual Peak Demand
    """
    # Ensure alignment
    # Pad to match df length exactly
    padding_len = len(df) - len(actions_continuous)
    if padding_len > 0:
        actions_continuous = actions_continuous + [0.0] * padding_len
    
    action_series = pd.Series(actions_continuous)[:len(df)]
    action_series.index = df.index
    action_series = action_series.fillna(0.0)
    
    # === BASELINE (no control) ===
    baseline_cooling = df['cooling_demand'].values
    baseline_heating = df['heating_demand'].values
    baseline_non_shiftable = df['non_shiftable_load'].values
    baseline_consumption = (baseline_cooling + baseline_heating + 
                           baseline_non_shiftable)
    
    baseline_carbon = (df['carbon_intensity'] * baseline_consumption).sum()
    baseline_cost = (df['electricity_pricing'] * baseline_consumption).sum()
    baseline_peak = np.max(baseline_consumption)
    
    # === CONTROLLED (with agent actions) ===
    controlled_cooling = df['cooling_demand'] * (1 + 0.1 * action_series)
    controlled_heating = df['heating_demand'] * (1 - 0.1 * action_series)
    controlled_consumption = (controlled_cooling + controlled_heating + 
                             df['non_shiftable_load']).values
    
    # Handle potential NaNs in controlled consumption
    if np.isnan(controlled_consumption).any():
        print(f"    WARNING: NaNs detected in controlled consumption for {building_name} ({label})")
        controlled_consumption = np.nan_to_num(controlled_consumption, nan=np.nanmean(controlled_consumption))
    
    controlled_carbon = (df['carbon_intensity'] * 
                        pd.Series(controlled_consumption, index=df.index)).sum()
    controlled_cost = (df['electricity_pricing'] * 
                      pd.Series(controlled_consumption, index=df.index)).sum()
    controlled_peak = np.max(controlled_consumption)
    
    # === METRIC 1-3: Carbon, Cost, Peak (existing) ===
    carbon_reduction = ((baseline_carbon - controlled_carbon) / 
                       baseline_carbon * 100) if baseline_carbon != 0 else 0.0
    cost_savings = ((baseline_cost - controlled_cost) / baseline_cost * 100) if baseline_cost != 0 else 0.0
    peak_reduction = ((baseline_peak - controlled_peak) / baseline_peak * 100) if baseline_peak != 0 else 0.0
    
    # === METRIC 4: RAMPING (Control Smoothness) ===
    # Calculate from action sequences
    if len(actions_discrete) > 1:
        action_diffs = np.abs(np.diff(actions_discrete))
        ramping = np.mean(action_diffs)
    else:
        ramping = 0.0
    
    # Baseline ramping (no control = no changes)
    baseline_ramping = 0.0
    
    # Normalized ramping (lower is better)
    # Since baseline is 0, we compare absolute values
    # Normalize to max possible ramping (4, from action 0 to 4)
    ramping_normalized = ramping / 4.0
    
    # === METRIC 5: LOAD FACTOR (Efficiency) ===
    # Load Factor = Average Load / Peak Load
    avg_consumption = np.mean(controlled_consumption)
    peak_consumption = np.max(controlled_consumption)
    load_factor = avg_consumption / peak_consumption if peak_consumption > 0 else 0
    
    # Baseline load factor
    baseline_avg = np.mean(baseline_consumption)
    baseline_peak_lf = np.max(baseline_consumption)
    baseline_load_factor = baseline_avg / baseline_peak_lf if baseline_peak_lf > 0 else 0
    
    # 1 - load_factor for table (lower is better)
    one_minus_lf = 1.0 - load_factor
    baseline_one_minus_lf = 1.0 - baseline_load_factor
    
    # Normalized (to baseline)
    load_factor_normalized = one_minus_lf / baseline_one_minus_lf if baseline_one_minus_lf != 0 else 1.0
    
    # === METRIC 6: AVERAGE PEAK DEMAND ===
    # Average of daily peak loads
    hours_per_day = 24
    num_days = len(controlled_consumption) // hours_per_day
    daily_peaks = []
    baseline_daily_peaks = []
    
    for day in range(num_days):
        start_idx = day * hours_per_day
        end_idx = start_idx + hours_per_day
        if end_idx <= len(controlled_consumption):
            daily_peaks.append(np.max(controlled_consumption[start_idx:end_idx]))
            baseline_daily_peaks.append(np.max(baseline_consumption[start_idx:end_idx]))
    
    avg_peak_demand = np.mean(daily_peaks) if daily_peaks else controlled_peak
    baseline_avg_peak = np.mean(baseline_daily_peaks) if baseline_daily_peaks else baseline_peak
    
    # Normalized to baseline
    avg_peak_normalized = avg_peak_demand / baseline_avg_peak if baseline_avg_peak > 0 else 1.0
    
    # === METRIC 7: ANNUAL PEAK DEMAND ===
    # Maximum peak across entire period
    annual_peak = controlled_peak
    baseline_annual_peak = baseline_peak
    
    # Normalized to baseline
    annual_peak_normalized = annual_peak / baseline_annual_peak if baseline_annual_peak > 0 else 1.0
    
    # === METRIC 8: NET ELECTRICITY CONSUMPTION ===
    net_consumption = np.sum(controlled_consumption)
    baseline_net_consumption = np.sum(baseline_consumption)
    
    # Normalized to baseline
    net_consumption_normalized = net_consumption / baseline_net_consumption if baseline_net_consumption > 0 else 1.0
    
    return {
        'building': building_name,
        'label': label,
        # Percentage improvements
        'carbon_reduction_pct': carbon_reduction,
        'cost_savings_pct': cost_savings,
        'peak_reduction_pct': peak_reduction,
        # Normalized values (for table, baseline=1.0, lower is better)
        'ramping_norm': ramping_normalized,
        'load_factor_norm': load_factor_normalized,
        'avg_peak_norm': avg_peak_normalized,
        'annual_peak_norm': annual_peak_normalized,
        'net_consumption_norm': net_consumption_normalized,
        # Raw values
        'ramping_raw': ramping,
        'load_factor_raw': load_factor,
        'avg_peak_raw': avg_peak_demand,
        'annual_peak_raw': annual_peak,
        'net_consumption_raw': net_consumption,
    }


def evaluate_metarl_agent():
    """Evaluate Recurrent Meta-RL agent"""
    print(f"\n Evaluating Meta-RL (Recurrent)...")
    
    agent = LSTMQAgent(INPUT_DIM_LSTM, HIDDEN_DIM_LSTM, OUTPUT_DIM)
    agent.model.load_state_dict(torch.load('../models/meta_rl_recurrent.pt'))
    agent.model.eval()
    
    all_tasks, scaler, columns = load_all_building_data(DATA_ROOT)
    test_tasks = all_tasks[4:6]
    
    results = []
    
    for i, df in enumerate(test_tasks):
        building_name = TEST_BUILDINGS[i]
        
        # Setup recurrent environment
        base_env = SimpleBuildingEnv(df, seq_len=SEQ_LEN, n_actions=OUTPUT_DIM, scaler=scaler, columns=columns)
        env = RecurrentBuildingEnv(base_env)
        
        # Get actions through environment interaction
        actions_discrete = []
        actions_continuous = []
        
        state_seq = env.reset()
        done = False
        while not done:
            state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0)
            q_vals = agent.predict(state_tensor).squeeze().detach().numpy()
            action = np.argmax(q_vals)
            cont_action = map_action_to_continuous(action)
            actions_discrete.append(action)
            actions_continuous.append(cont_action)
            state_seq, _, done, _ = env.step(action)
        
        # Calculate all metrics
        metrics = calculate_comprehensive_metrics(
            df, actions_continuous, actions_discrete, 'Meta-RL', building_name
        )
        results.append(metrics)
        
        print(f"  {building_name}:")
        print(f"    Cost: {metrics['cost_savings_pct']:.2f}%")
        print(f"    Ramping: {metrics['ramping_norm']:.3f}")
        print(f"    Load Factor: {metrics['load_factor_norm']:.3f}")
    
    return results


def evaluate_sac_agent():
    """Evaluate SAC agent"""
    print(f"\nEvaluating Standard RL (SAC)...")
    
    action_space = ActionSpace(np.array([-1.0]), np.array([1.0]), (1,))
    agent = SACAgent(INPUT_DIM_SAC, action_space, hidden_dim=HIDDEN_DIM_SAC)
    agent.load_model('../models/sac_model.pt')
    
    all_tasks, scaler, columns = load_all_building_data(DATA_ROOT)
    test_tasks = all_tasks[4:6]
    
    results = []
    
    for i, df in enumerate(test_tasks):
        building_name = TEST_BUILDINGS[i]
        
        # Setup environment
        env = SimpleBuildingEnv(df, seq_len=SEQ_LEN, n_actions=OUTPUT_DIM, scaler=scaler, columns=columns)
        
        # Get actions through environment interaction
        actions_discrete = []
        actions_continuous = []
        
        state_seq = env.reset()
        state_flat = state_seq.flatten()
        done = False
        while not done:
            action = agent.select_action(state_flat, evaluate=True)
            actions_continuous.append(action[0])
            # Convert continuous action to discrete for ramping calculation
            # Map from [-1, 1] back to [0, 4]
            discrete_action = int((action[0] + 1.0) * (OUTPUT_DIM - 1) / 2.0)
            discrete_action = np.clip(discrete_action, 0, OUTPUT_DIM - 1)
            actions_discrete.append(discrete_action)
            next_state_seq, _, done, _ = env.step(action[0])
            state_flat = next_state_seq.flatten()
        
        # Calculate all metrics
        metrics = calculate_comprehensive_metrics(
            df, actions_continuous, actions_discrete, 'Standard RL', building_name
        )
        results.append(metrics)
        
        print(f"  {building_name}:")
        print(f"    Cost: {metrics['cost_savings_pct']:.2f}%")
        print(f"    Ramping: {metrics['ramping_norm']:.3f}")
        print(f"    Load Factor: {metrics['load_factor_norm']:.3f}")
    
    return results


def evaluate_rbc_agent():
    """Evaluate RBC agent"""
    print(f"\nEvaluating RBC...")
    
    agent = RBCAgent()
    all_tasks, scaler, columns = load_all_building_data(DATA_ROOT)
    test_tasks = all_tasks[4:6]
    
    results = []
    
    for i, df in enumerate(test_tasks):
        building_name = TEST_BUILDINGS[i]
        
        # Get actions
        actions_discrete = []
        actions_continuous = []
        
        for j in range(len(df) - SEQ_LEN):
            state = df.iloc[j + SEQ_LEN].values
            action = agent.predict(state)
            cont_action = map_action_to_continuous(action)
            actions_discrete.append(action)
            actions_continuous.append(cont_action)
        
        # Calculate all metrics
        metrics = calculate_comprehensive_metrics(
            df, actions_continuous, actions_discrete, 'RBC', building_name
        )
        results.append(metrics)
        
        print(f"  {building_name}:")
        print(f"    Cost: {metrics['cost_savings_pct']:.2f}%")
        print(f"    Ramping: {metrics['ramping_norm']:.3f}")
        print(f"    Load Factor: {metrics['load_factor_norm']:.3f}")
    
    return results


# Evaluate all three methods
print("\n" + "="*80)
print("EVALUATING ALL METHODS")
print("="*80)

meta_results = evaluate_metarl_agent()
rl_results = evaluate_sac_agent()
rbc_results = evaluate_rbc_agent()

# Calculate averages
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
    
    print(f"\n{method_name}:")
    print(f"  Cost Savings: {avg_metrics['Cost_%']:.2f}%")
    print(f"  Ramping (norm): {avg_metrics['Ramping']:.3f}")
    print(f"  Load Factor (norm): {avg_metrics['Load_Factor']:.3f}")
    print(f"  Avg Peak (norm): {avg_metrics['Avg_Peak']:.3f}")
    print(f"  Annual Peak (norm): {avg_metrics['Annual_Peak']:.3f}")
    print(f"  Net Consumption (norm): {avg_metrics['Net_Consumption']:.3f}")

# Save results
summary_df = pd.DataFrame(summary)
summary_df.to_csv(f'{OUTPUT_DIR}/all_metrics.csv', index=False)

print(f"\n Saved: {OUTPUT_DIR}/all_metrics.csv")

# Calculate improvements (Meta-RL vs RBC)
print("\n" + "="*80)
print("META-RL IMPROVEMENT OVER RBC")
print("="*80)

meta_row = summary_df[summary_df['Method'] == 'Meta-RL'].iloc[0]
rbc_row = summary_df[summary_df['Method'] == 'RBC'].iloc[0]

improvements = {
    'Ramping': ((rbc_row['Ramping'] - meta_row['Ramping']) / rbc_row['Ramping'] * 100),
    'Load_Factor': ((rbc_row['Load_Factor'] - meta_row['Load_Factor']) / rbc_row['Load_Factor'] * 100),
    'Avg_Peak': ((rbc_row['Avg_Peak'] - meta_row['Avg_Peak']) / rbc_row['Avg_Peak'] * 100),
    'Annual_Peak': ((rbc_row['Annual_Peak'] - meta_row['Annual_Peak']) / rbc_row['Annual_Peak'] * 100),
    'Net_Consumption': ((rbc_row['Net_Consumption'] - meta_row['Net_Consumption']) / rbc_row['Net_Consumption'] * 100),
}

for metric, improvement in improvements.items():
    print(f"  {metric}: {improvement:.2f}%")

print("\n" + "="*80)
print(" ALL METRICS CALCULATED FROM REAL DATA!")
print("="*80)
