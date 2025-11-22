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


import os
import torch
import pandas as pd
import numpy as np
from lstm_q_agent import LSTMQAgent
from rbc_agent import RBCAgent
from agent_setup import load_all_building_data

# Config
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
print("METRICS EVALUATION")
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
    
    # === BASELINE (no control) ===
    baseline_cooling = df['cooling_demand'].values
    baseline_heating = df['heating_demand'].values
    baseline_non_shiftable = df['non_shiftable_load'].values
    baseline_consumption = (baseline_cooling + baseline_heating + 
                           baseline_non_shiftable)
    
    baseline_carbon = (df['carbon_intensity'] * baseline_consumption).sum()
    baseline_cost = (df['electricity_pricing'] * baseline_consumption).sum()
    baseline_peak = baseline_consumption.max()
    
    # === CONTROLLED (with agent actions) ===
    action_series = pd.Series(actions_continuous + [0.0] * SEQ_LEN)[:len(df)]
    
    controlled_cooling = df['cooling_demand'] * (1 + 0.1 * action_series)
    controlled_heating = df['heating_demand'] * (1 - 0.1 * action_series)
    controlled_consumption = (controlled_cooling + controlled_heating + 
                             df['non_shiftable_load']).values
    
    controlled_carbon = (df['carbon_intensity'] * 
                        pd.Series(controlled_consumption)).sum()
    controlled_cost = (df['electricity_pricing'] * 
                      pd.Series(controlled_consumption)).sum()
    controlled_peak = controlled_consumption.max()
    
    # === METRIC 1-3: Carbon, Cost, Peak (existing) ===
    carbon_reduction = ((baseline_carbon - controlled_carbon) / 
                       baseline_carbon * 100)
    cost_savings = ((baseline_cost - controlled_cost) / baseline_cost * 100)
    peak_reduction = ((baseline_peak - controlled_peak) / baseline_peak * 100)
    
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
    baseline_load_factor = baseline_avg / baseline_peak_lf
    
    # 1 - load_factor for table (lower is better)
    one_minus_lf = 1.0 - load_factor
    baseline_one_minus_lf = 1.0 - baseline_load_factor
    
    # Normalized (to baseline)
    load_factor_normalized = one_minus_lf / baseline_one_minus_lf
    
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
    avg_peak_normalized = avg_peak_demand / baseline_avg_peak
    
    # === METRIC 7: ANNUAL PEAK DEMAND ===
    # Maximum peak across entire period
    annual_peak = controlled_peak
    baseline_annual_peak = baseline_peak
    
    # Normalized to baseline
    annual_peak_normalized = annual_peak / baseline_annual_peak
    
    # === METRIC 8: NET ELECTRICITY CONSUMPTION ===
    net_consumption = np.sum(controlled_consumption)
    baseline_net_consumption = np.sum(baseline_consumption)
    
    # Normalized to baseline
    net_consumption_normalized = net_consumption / baseline_net_consumption
    
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


def evaluate_lstm_agent(agent_path, label):
    """Evaluate LSTM agent (Meta-RL or Standard RL)"""
    print(f"\nEvaluating {label}...")
    
    agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    agent.model.load_state_dict(torch.load(agent_path))
    agent.model.eval()
    
    all_tasks = load_all_building_data(DATA_ROOT)
    test_tasks = all_tasks[4:6]
    
    results = []
    
    for i, df in enumerate(test_tasks):
        building_name = TEST_BUILDINGS[i]
        
        # Get sequences
        sequences = []
        for j in range(len(df) - SEQ_LEN):
            seq = df.iloc[j:j+SEQ_LEN].values
            sequences.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(0))
        
        # Get actions
        actions_discrete = []
        actions_continuous = []
        
        for seq in sequences:
            q_vals = agent.predict(seq).squeeze().detach().numpy()
            action = np.argmax(q_vals)
            cont_action = map_action_to_continuous(action)
            actions_discrete.append(action)
            actions_continuous.append(cont_action)
        
        # Calculate all metrics
        metrics = calculate_comprehensive_metrics(
            df, actions_continuous, actions_discrete, label, building_name
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
    all_tasks = load_all_building_data(DATA_ROOT)
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

meta_results = evaluate_lstm_agent('../models/meta_rl_model.pt', 'Meta-RL')
rl_results = evaluate_lstm_agent('../models/rl_model.pt', 'Standard RL')
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

print(f"\nSaved: {OUTPUT_DIR}/all_metrics.csv")

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