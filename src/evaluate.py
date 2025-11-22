#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 21:42:11 2025

@author: ivanovsi
"""

# ----------------------------------------
# WP3: Compare Standard RL vs Meta-RL Agent
# Evaluate on UNSEEN buildings (5-6) with proper KPIs
# ----------------------------------------

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data
from reward_utils import compute_reward_single

# Config
DATA_ROOT = '../outputs/phase3'
OUTPUT_DIR = '../outputs/eval/test_buildings'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 17
HIDDEN_DIM = 64
OUTPUT_DIM = 5
SEQ_LEN = 8
# Evaluate ONLY on test buildings (5-6) that were NOT seen during training
TEST_BUILDINGS = [f'Building_{i+1}' for i in range(4, 6)]  # Buildings 5 and 6
EPSILON = 0.0  # No exploration during evaluation
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

print(f" Evaluation on TEST buildings: {TEST_BUILDINGS}")
print(f"   (Training was on Building_1 to Building_4)")

def create_sequences(df, seq_len=8):
    """Convert a flat time-series DataFrame into LSTM input sequences."""
    X = []
    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i+seq_len].values
        X.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(0))
    return X

def reward_proxy(row, action):
    """Simplified reward proxy matching training objective."""
    # Using unified reward function - action parameter kept for compatibility
    # but not used to match training (no action-dependent terms in training)
    return compute_reward_single(row)

def map_action_to_continuous(action_idx):
    """Map discrete action (0 to 4) to continuous range [-1.0, 1.0]."""
    return -1.0 + 2.0 * action_idx / (OUTPUT_DIM - 1)

def check_model_weights(rl_path, meta_rl_path):
    """Verify model weights are distinct."""
    rl_weights = torch.load(rl_path)
    meta_rl_weights = torch.load(meta_rl_path)
    total_diff = 0.0
    print("\nModel Weight Differences:")
    for key in rl_weights:
        diff = torch.norm(rl_weights[key] - meta_rl_weights[key]).item()
        total_diff += diff
        print(f"  {key}: {diff:.6f}")
    print(f"Total weight difference: {total_diff:.6f}")
    if total_diff < 1e-3:
        print("⚠️ Error: Models are identical or nearly identical!")
        return False
    return True

def compute_kpis(df, actions_continuous, label, building_name):
    """
    Compute KPIs: carbon emissions, energy cost, peak load.
    Actions affect cooling/heating demands and electricity usage.
    """
    # Baseline (no control action)
    baseline_carbon = (df['carbon_intensity'] * 
                      (df['cooling_demand'] + df['heating_demand'] + 
                       df['non_shiftable_load'])).sum()
    baseline_cost = (df['electricity_pricing'] * 
                    (df['cooling_demand'] + df['heating_demand'] + 
                     df['non_shiftable_load'])).sum()
    baseline_peak = (df['cooling_demand'] + df['heating_demand'] + 
                     df['non_shiftable_load']).max()
    
    # With control actions (agent modulates demand)
    # Action affects demand: positive action increases, negative decreases
    action_series = pd.Series(actions_continuous + [0.0] * SEQ_LEN)[:len(df)]
    controlled_cooling = df['cooling_demand'] * (1 + 0.1 * action_series)
    controlled_heating = df['heating_demand'] * (1 - 0.1 * action_series)
    total_load = controlled_cooling + controlled_heating + df['non_shiftable_load']
    
    controlled_carbon = (df['carbon_intensity'] * total_load).sum()
    controlled_cost = (df['electricity_pricing'] * total_load).sum()
    controlled_peak = total_load.max()
    
    # Calculate reductions (negative means worse)
    carbon_reduction = ((baseline_carbon - controlled_carbon) / 
                       baseline_carbon * 100)
    cost_savings = ((baseline_cost - controlled_cost) / baseline_cost * 100)
    peak_reduction = ((baseline_peak - controlled_peak) / baseline_peak * 100)
    
    return {
        'building': building_name,
        'label': label,
        'carbon_emissions': controlled_carbon,
        'carbon_reduction_pct': carbon_reduction,
        'energy_cost': controlled_cost,
        'cost_savings_pct': cost_savings,
        'peak_load': controlled_peak,
        'peak_reduction_pct': peak_reduction,
        'baseline_carbon': baseline_carbon,
        'baseline_cost': baseline_cost,
        'baseline_peak': baseline_peak
    }


def evaluate_agent(agent_weights_path, label):
    print(f"\nEvaluating {label}...\n")
    agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    agent.model.load_state_dict(torch.load(agent_weights_path))
    agent.model.eval()

    # Load all buildings but use only test buildings (5-6)
    all_task_data = load_all_building_data(DATA_ROOT)
    task_data = all_task_data[4:6]  # Buildings 5 and 6
    results = []

    for i, df in enumerate(task_data):
        building_name = TEST_BUILDINGS[i]
        sequences = create_sequences(df, SEQ_LEN)

        actions = []
        q_values = []
        continuous_actions = []
        rewards = []

        for j, seq in enumerate(sequences):
            q_vals = agent.predict(seq).squeeze().numpy()
            if np.random.random() < EPSILON:
                action = np.random.randint(0, OUTPUT_DIM)
            else:
                action = np.argmax(q_vals)
            cont_action = map_action_to_continuous(action)
            reward = reward_proxy(df.iloc[j+SEQ_LEN], cont_action)
            actions.append(action)
            q_values.append(q_vals)
            continuous_actions.append(cont_action)
            rewards.append(reward)
            # Debug: Print first few rewards
            if j < 5:
                print(f"{building_name} ({label}) Timestep {j}: Action = {action}, Cont Action = {cont_action:.3f}, Reward = {reward:.3f}")

        avg_reward = np.mean(rewards) if rewards else 0.0
        q_std = np.std(q_values) if q_values else 0.0
        
        # Compute KPIs
        kpis = compute_kpis(df, continuous_actions, label, building_name)

        results.append({
            'building': building_name,
            'avg_reward': avg_reward,
            'actions': actions,
            'q_values': q_values,
            'q_std': q_std,
            'continuous_actions': continuous_actions,
            'rewards': rewards,
            'kpis': kpis
        })

        # Plot rewards
        plt.figure(figsize=(12, 4))
        plt.plot(rewards, label='Proxy Reward')
        plt.title(f'{building_name} - Reward Over Time ({label})')
        plt.xlabel('Timestep')
        plt.ylabel('Reward')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 
                                 f'{building_name}_{label}_reward.png'))
        plt.close()

        # Plot continuous actions
        plt.figure(figsize=(12, 4))
        plt.plot(continuous_actions, label='Continuous Action')
        plt.title(f'{building_name} - Continuous Actions ({label})')
        plt.xlabel('Timestep')
        plt.ylabel('Action Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, 
                                 f'{building_name}_{label}_cont_action.png'))
        plt.close()

        print(f"{building_name} ({label}):")
        print(f"   Avg Reward: {avg_reward:.3f}")
        print(f"   Carbon Reduction: {kpis['carbon_reduction_pct']:.2f}%")
        print(f"   Cost Savings: {kpis['cost_savings_pct']:.2f}%")
        print(f"   Peak Reduction: {kpis['peak_reduction_pct']:.2f}%\n")

    return results

# Verify model weights
print("Verifying model weights...")
if not check_model_weights('../models/rl_model.pt', '../models/meta_rl_model.pt'):
    exit(1)

# Evaluate both agents
rl_results = evaluate_agent('../models/rl_model.pt', 'RL')
meta_rl_results = evaluate_agent('../models/meta_rl_model.pt', 'MetaRL')

# Compare results with KPIs
print("\n" + "="*80)
print(" KPI COMPARISON: Meta-RL vs Standard RL (on TEST buildings 5-6)")
print("="*80)

kpi_comparison = []
for i, building in enumerate(TEST_BUILDINGS):
    rl_kpis = rl_results[i]['kpis']
    meta_kpis = meta_rl_results[i]['kpis']
    
    print(f"\n{building}:")
    print(f"   {'Metric':<25} {'RL':<15} {'Meta-RL':<15} {'Diff':<10}")
    print(f"   {'-'*65}")
    print(f"   {'Carbon Reduction %':<25} {rl_kpis['carbon_reduction_pct']:>14.2f} "
          f"{meta_kpis['carbon_reduction_pct']:>14.2f} "
          f"{meta_kpis['carbon_reduction_pct']-rl_kpis['carbon_reduction_pct']:>9.2f}")
    print(f"   {'Cost Savings %':<25} {rl_kpis['cost_savings_pct']:>14.2f} "
          f"{meta_kpis['cost_savings_pct']:>14.2f} "
          f"{meta_kpis['cost_savings_pct']-rl_kpis['cost_savings_pct']:>9.2f}")
    print(f"   {'Peak Reduction %':<25} {rl_kpis['peak_reduction_pct']:>14.2f} "
          f"{meta_kpis['peak_reduction_pct']:>14.2f} "
          f"{meta_kpis['peak_reduction_pct']-rl_kpis['peak_reduction_pct']:>9.2f}")
    
    kpi_comparison.append({
        'Building': building,
        'RL_Carbon_Reduction_%': rl_kpis['carbon_reduction_pct'],
        'MetaRL_Carbon_Reduction_%': meta_kpis['carbon_reduction_pct'],
        'RL_Cost_Savings_%': rl_kpis['cost_savings_pct'],
        'MetaRL_Cost_Savings_%': meta_kpis['cost_savings_pct'],
        'RL_Peak_Reduction_%': rl_kpis['peak_reduction_pct'],
        'MetaRL_Peak_Reduction_%': meta_kpis['peak_reduction_pct'],
        'RL_Emissions': rl_kpis['carbon_emissions'],
        'MetaRL_Emissions': meta_kpis['carbon_emissions'],
        'RL_Cost': rl_kpis['energy_cost'],
        'MetaRL_Cost': meta_kpis['energy_cost'],
        'RL_Peak': rl_kpis['peak_load'],
        'MetaRL_Peak': meta_kpis['peak_load']
    })

# Save KPI comparison
kpi_df = pd.DataFrame(kpi_comparison)
kpi_df.to_csv(os.path.join(OUTPUT_DIR, 'kpi_comparison.csv'), index=False)

# Calculate AVERAGE improvements
avg_carbon_rl = np.mean([r['kpis']['carbon_reduction_pct'] for r in rl_results])
avg_carbon_meta = np.mean([r['kpis']['carbon_reduction_pct'] 
                           for r in meta_rl_results])
avg_cost_rl = np.mean([r['kpis']['cost_savings_pct'] for r in rl_results])
avg_cost_meta = np.mean([r['kpis']['cost_savings_pct'] 
                         for r in meta_rl_results])
avg_peak_rl = np.mean([r['kpis']['peak_reduction_pct'] for r in rl_results])
avg_peak_meta = np.mean([r['kpis']['peak_reduction_pct'] 
                         for r in meta_rl_results])

print("\n" + "="*80)
print("FINAL SUMMARY: Meta-RL Performance vs Standard RL")
print("="*80)
print(f"\nThe Meta-RL method achieves:")
print(f"  • {avg_carbon_meta:.2f}% reduction in carbon emissions")
print(f"    (Standard RL: {avg_carbon_rl:.2f}% | Improvement: "
      f"{avg_carbon_meta - avg_carbon_rl:+.2f}%)")
print(f"  • {avg_cost_meta:.2f}% improvement in energy cost savings")
print(f"    (Standard RL: {avg_cost_rl:.2f}% | Improvement: "
      f"{avg_cost_meta - avg_cost_rl:+.2f}%)")
print(f"  • {avg_peak_meta:.2f}% reduction in peak electricity loads")
print(f"    (Standard RL: {avg_peak_rl:.2f}% | Improvement: "
      f"{avg_peak_meta - avg_peak_rl:+.2f}%)")
print("\ncompared to standard RL approach.")
print("="*80)



print(f"\n Results saved to '{OUTPUT_DIR}/'")
print(f"   - kpi_comparison.csv")
print(f"   - Plots for each building")
