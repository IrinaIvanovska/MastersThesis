#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 13:35:05 2025

@author: ivanovsi
"""

# ----------------------------------------
# Compare Standard RL vs Meta-RL Agent
# ----------------------------------------

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data

# Config
DATA_ROOT = 'outputs/phase3'
OUTPUT_DIR = 'outputs/both'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 28
HIDDEN_DIM = 64
OUTPUT_DIM = 5
SEQ_LEN = 8
BUILDINGS = [f'Building_{i+1}' for i in range(6)]
EPSILON = 0.2
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def create_sequences(df, seq_len=8):
    """Convert a flat time-series DataFrame into LSTM input sequences."""
    X = []
    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i+seq_len].values
        X.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(0))
    return X

def reward_proxy(row, action):
    """Action-dependent reward proxy for both RL and Meta-RL."""
    base_reward = -(
        0.7 * abs(float(row['average_unmet_cooling_setpoint_difference'])) +
        0.3 * float(row['cooling_demand']) +
        0.3 * float(row['heating_demand']) +
        0.1 * abs(float(row['solar_generation'])) +
        0.05 * float(row['non_shiftable_load'])
    )
    action_penalty = 0.1 * (action ** 2)
    demand_adjustment = 0.2 * action * float(row['cooling_demand'])
    setpoint_adjustment = 0.1 * action * abs(float(row['average_unmet_cooling_setpoint_difference']))
    return base_reward - action_penalty - demand_adjustment + setpoint_adjustment

def map_action_to_continuous(action_idx):
    """Map discrete action (0 to 4) to continuous range [-1.0, 1.0]."""
    return -1.0 + 2.0 * action_idx / (OUTPUT_DIM - 1)

def check_model_weights(rl_path, meta_rl_path):
    """Verify model weights are distinct."""
    rl_weights = torch.load(rl_path)
    meta_rl_weights = torch.load(meta_rl_path)
    total_diff = 0.0
    print("\n🔍 Model Weight Differences:")
    for key in rl_weights:
        diff = torch.norm(rl_weights[key] - meta_rl_weights[key]).item()
        total_diff += diff
        print(f"  {key}: {diff:.6f}")
    print(f"Total weight difference: {total_diff:.6f}")
    if total_diff < 1e-3:
        print("⚠️ Error: Models are identical or nearly identical!")
        return False
    return True

def compute_all_kpis(df):
    # Ensure datetime index
    if not np.issubdtype(df.index.dtype, np.datetime64):
        df = df.copy()
        df["datetime"] = pd.date_range(start="2025-01-01", periods=len(df), freq="H")
        df.set_index("datetime", inplace=True)

    # Default fallback if a column is missing
    def safe_col(col):
        return df[col] if col in df.columns else pd.Series(np.zeros(len(df)), index=df.index)

    electricity_demand = safe_col("electricity_demand")
    unserved_energy = safe_col("unserved_energy")
    carbon_intensity = safe_col("carbon_intensity")
    electricity_pricing = safe_col("electricity_pricing")
    solar_generation = safe_col("solar_generation")
    cold_diff = safe_col("average_unmet_heating_setpoint_difference")
    hot_diff = safe_col("average_unmet_cooling_setpoint_difference")
    power_outage = safe_col("power_outage")

    # Daily/Monthly aggregates
    daily = electricity_demand.resample("D")
    monthly = electricity_demand.resample("M")

    # KPIs
    kpis = {
        "all_time_peak_average": electricity_demand.max(),
        "annual_normalized_unserved_energy_total": unserved_energy.sum() / electricity_demand.sum() if electricity_demand.sum() > 0 else 0,
        "carbon_emissions_total": (electricity_demand * carbon_intensity).sum(),
        "cost_total": (electricity_demand * electricity_pricing).sum(),
        "daily_one_minus_load_factor_average": (1 - (daily.mean() / daily.max())).mean(),
        "daily_peak_average": daily.max().mean(),
        "discomfort_cold_delta_average": cold_diff.mean(),
        "discomfort_cold_delta_maximum": cold_diff.max(),
        "discomfort_cold_delta_minimum": cold_diff.min(),
        "discomfort_cold_proportion": (cold_diff > 0).mean(),
        "discomfort_hot_delta_average": hot_diff.mean(),
        "discomfort_hot_delta_maximum": hot_diff.max(),
        "discomfort_hot_delta_minimum": hot_diff.min(),
        "discomfort_hot_proportion": (hot_diff > 0).mean(),
        "discomfort_proportion": ((cold_diff > 0) | (hot_diff > 0)).mean(),
        "electricity_consumption_total": electricity_demand.sum(),
        "monthly_one_minus_load_factor_average": (1 - (monthly.mean() / monthly.max())).mean(),
        "one_minus_thermal_resilience_proportion": ((cold_diff > 1) | (hot_diff > 1)).mean(),
        "power_outage_normalized_unserved_energy_total": unserved_energy[power_outage == 1].sum() / electricity_demand.sum() if electricity_demand.sum() > 0 else 0,
        "ramping_average": np.abs(electricity_demand.diff()).mean(),
        "zero_net_energy": (electricity_demand - solar_generation).sum()
    }

    return kpis


def evaluate_agent(agent_weights_path, label):
    print(f"\n📥 Evaluating {label}...")
    agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    agent.model.load_state_dict(torch.load(agent_weights_path))
    agent.model.eval()

    task_data = load_all_building_data(DATA_ROOT)
    results = []

    for i, df in enumerate(task_data):
        building_name = BUILDINGS[i]
        sequences = create_sequences(df, SEQ_LEN)

        actions = []
        q_values = []
        continuous_actions = []
        rewards = []
        adjusted_demand = df['electricity_demand'].copy()


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
            adjusted_demand.iloc[j+SEQ_LEN] += continuous_actions[j]  # Or use your own formula

            # Debug: Print first few rewards
            if j < 5:
                print(f"{building_name} ({label}) Timestep {j}: Action = {action}, Cont Action = {cont_action:.3f}, Reward = {reward:.3f}")

        df['electricity_demand'] = adjusted_demand
        
        avg_reward = np.mean(rewards) if rewards else 0.0
        q_std = np.std(q_values) if q_values else 0.0
        
        kpis = compute_all_kpis(df)

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
        plt.savefig(os.path.join(OUTPUT_DIR, f'{building_name}_{label}_reward.png'))
        plt.close()

        # Plot action distribution
        plt.figure(figsize=(12, 4))
        plt.hist(actions, bins=OUTPUT_DIM, range=(0, OUTPUT_DIM), align='left', rwidth=0.8)
        plt.title(f'{building_name} - Action Distribution ({label})')
        plt.xlabel('Action')
        plt.ylabel('Frequency')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{building_name}_{label}_action_dist.png'))
        plt.close()

        # Plot continuous actions
        plt.figure(figsize=(12, 4))
        plt.plot(continuous_actions, label='Continuous Action')
        plt.title(f'{building_name} - Continuous Actions ({label})')
        plt.xlabel('Timestep')
        plt.ylabel('Action Value')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f'{building_name}_{label}_cont_action.png'))
        plt.close()

        print(f"🏢 {building_name} - Avg Reward ({label}): {avg_reward:.3f}, Q-Value Std: {q_std:.3f}")

    return results

# Verify model weights
print("🔍 Verifying model weights...")
if not check_model_weights('rl_model.pt', 'meta_rl_model.pt'):
    exit(1)

# Evaluate both agents
rl_results = evaluate_agent('rl_model.pt', 'RL')
meta_rl_results = evaluate_agent('meta_rl_model.pt', 'MetaRL')

# Compare results
print("\n📊 Comparison Summary:")
comparison_data = []
for i, building in enumerate(BUILDINGS):
    rl_avg = rl_results[i]['avg_reward']
    meta_rl_avg = meta_rl_results[i]['avg_reward']
    diff = meta_rl_avg - rl_avg
    rl_q_std = rl_results[i]['q_std']
    meta_rl_q_std = meta_rl_results[i]['q_std']
    action_diff = np.mean(np.array(rl_results[i]['actions']) != np.array(meta_rl_results[i]['actions']))
    reward_std = np.std(rl_results[i]['rewards'] + meta_rl_results[i]['rewards'])
    
    rl_kpis = rl_results[i]['kpis']
    meta_kpis = meta_rl_results[i]['kpis']
    
    comparison_data.append([
        building,
        rl_avg, meta_rl_avg, diff,
        rl_q_std, meta_rl_q_std,
        action_diff, reward_std,
        rl_kpis['cost_total'], meta_kpis['cost_total'], meta_kpis['cost_total'] - rl_kpis['cost_total'],
        rl_kpis['carbon_emissions_total'], meta_kpis['carbon_emissions_total'], meta_kpis['carbon_emissions_total'] - rl_kpis['carbon_emissions_total'],
        rl_kpis['discomfort_hot_delta_average'], meta_kpis['discomfort_hot_delta_average'], meta_kpis['discomfort_hot_delta_average'] - rl_kpis['discomfort_hot_delta_average'],
        rl_kpis['discomfort_proportion'], meta_kpis['discomfort_proportion'], meta_kpis['discomfort_proportion'] - rl_kpis['discomfort_proportion'],
    ])    
    print(f"{building}: RL Avg Reward = {rl_avg:.3f}, Meta-RL Avg Reward = {meta_rl_avg:.3f}, "
          f"Diff = {diff:.3f}, RL Q-Std = {rl_q_std:.3f}, Meta-RL Q-Std = {meta_rl_q_std:.3f}, "
          f"Action Diff = {action_diff:.3f}, Reward Std = {reward_std:.3f}, "
          f"RL Cost = {rl_kpis['cost_total']:.2f}, MetaRL Cost = {meta_kpis['cost_total']:.2f}, "
          f"RL Emissions = {rl_kpis['carbon_emissions_total']:.2f}, MetaRL Emissions = {meta_kpis['carbon_emissions_total']:.2f}")


comparison_df = pd.DataFrame([{
    'Building': BUILDINGS[i],
    'RL_Avg_Reward': rl_results[i]['avg_reward'],
    'MetaRL_Avg_Reward': meta_rl_results[i]['avg_reward'],
    'Difference': meta_rl_results[i]['avg_reward'] - rl_results[i]['avg_reward'],
    'RL_Q_Std': rl_results[i]['q_std'],
    'MetaRL_Q_Std': meta_rl_results[i]['q_std'],
    'Action_Difference': np.mean(np.array(rl_results[i]['actions']) != np.array(meta_rl_results[i]['actions'])),
    'Reward_Std': np.std(rl_results[i]['rewards'] + meta_rl_results[i]['rewards']),
    'RL_Cost': rl_results[i]['kpis']['cost_total'],
    'MetaRL_Cost': meta_rl_results[i]['kpis']['cost_total'],
    'Cost_Diff': meta_rl_results[i]['kpis']['cost_total'] - rl_results[i]['kpis']['cost_total'],
    'RL_Emissions': rl_results[i]['kpis']['carbon_emissions_total'],
    'MetaRL_Emissions': meta_rl_results[i]['kpis']['carbon_emissions_total'],
    'Emissions_Diff': meta_rl_results[i]['kpis']['carbon_emissions_total'] - rl_results[i]['kpis']['carbon_emissions_total']
} for i in range(len(BUILDINGS))])

comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'comparison_summary.csv'), index=False)
print(f"\n✅ Comparison summary saved to {os.path.join(OUTPUT_DIR, 'comparison_summary.csv')}")

q_stats = []
for i, building in enumerate(BUILDINGS):
    q_stats.append({
        'Building': building,
        'RL_Q_Mean': np.mean(rl_results[i]['q_values']),
        'RL_Q_Std': rl_results[i]['q_std'],
        'MetaRL_Q_Mean': np.mean(meta_rl_results[i]['q_values']),
        'MetaRL_Q_Std': meta_rl_results[i]['q_std']
    })
q_stats_df = pd.DataFrame(q_stats)
q_stats_df.to_csv(os.path.join(OUTPUT_DIR, 'q_value_stats.csv'), index=False)
print(f"✅ Q-value stats saved to {os.path.join(OUTPUT_DIR, 'q_value_stats.csv')}")

print("\n✅ Plots saved in 'outputs/both/'")

# --- Plot summary comparison of average rewards ---
def plot_comparison(buildings, rl_rewards, meta_rl_rewards, save_path=None):
    x = np.arange(len(buildings))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, rl_rewards, width, label='RL', color='orange')
    plt.bar(x + width/2, meta_rl_rewards, width, label='Meta-RL', color='steelblue')

    plt.xticks(x, buildings, rotation=45)
    plt.ylabel('Average Reward')
    plt.title('Average Reward Comparison (Meta-RL vs RL)')
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print(f"✅ Reward comparison plot saved to {save_path}")
    plt.show()

plot_comparison(
    buildings=BUILDINGS,
    rl_rewards=[r['avg_reward'] for r in rl_results],
    meta_rl_rewards=[r['avg_reward'] for r in meta_rl_results],
    save_path=os.path.join(OUTPUT_DIR, 'reward_comparison.png')
)

