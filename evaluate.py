#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 13:35:05 2025

@author: ivanovsi
"""

# ----------------------------------------
# WP3: Compare Standard RL vs Meta-RL Agent
# ----------------------------------------

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lstm_q_agent import LSTMQAgent
from wp2_agent_setup import load_all_building_data

# Config
DATA_ROOT = 'outputs/wp1_phase3'
OUTPUT_DIR = 'outputs/wp3/both'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 15
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

        results.append({
            'building': building_name,
            'avg_reward': avg_reward,
            'actions': actions,
            'q_values': q_values,
            'q_std': q_std,
            'continuous_actions': continuous_actions,
            'rewards': rewards
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
    comparison_data.append([building, rl_avg, meta_rl_avg, diff, rl_q_std, meta_rl_q_std, action_diff, reward_std])
    print(f"{building}: RL Avg Reward = {rl_avg:.3f}, Meta-RL Avg Reward = {meta_rl_avg:.3f}, "
          f"Diff = {diff:.3f}, RL Q-Std = {rl_q_std:.3f}, Meta-RL Q-Std = {meta_rl_q_std:.3f}, "
          f"Action Diff = {action_diff:.3f}, Reward Std = {reward_std:.3f}")

# Save comparison to CSV
comparison_df = pd.DataFrame(comparison_data, columns=[
    'Building', 'RL_Avg_Reward', 'MetaRL_Avg_Reward', 'Difference',
    'RL_Q_Std', 'MetaRL_Q_Std', 'Action_Difference', 'Reward_Std'
])
comparison_df.to_csv(os.path.join(OUTPUT_DIR, 'comparison_summary.csv'), index=False)
print(f"\n✅ Comparison summary saved to {os.path.join(OUTPUT_DIR, 'comparison_summary.csv')}")

# Save Q-value stats
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

print("\n✅ WP3 complete. Plots saved in 'outputs/wp3/both/'")
