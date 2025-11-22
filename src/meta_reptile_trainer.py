#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 20:21:16 2025

@author: ivanovsi
"""

# ----------------------------------------
# Reptile Meta-RL Trainer using LSTM-Q Agent
# ----------------------------------------

import torch
import random
import numpy as np
from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data
from reward_utils import compute_reward_with_action

# Config
INPUT_DIM = 17
HIDDEN_DIM = 64
OUTPUT_DIM = 5
INNER_STEPS = 10
META_EPISODES = 400  # Increased from 100 for better convergence
META_LR = 1.0
SEQ_LEN = 8
SEED = 2718  # Set random seed for reproducibility

# Set all random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Load building data (training set only: buildings 1-4)
all_tasks = load_all_building_data('../outputs/phase3')
tasks = all_tasks[:4]  # Use only first 4 buildings for training
print(f"Training Meta-RL on {len(tasks)} buildings (Building_1 to Building_4)")

# Create action-aware training sequences for Meta-RL
def create_action_aware_sequences(df, seq_len=8, n_actions=5):
    """
    Create training data with action-dependent rewards.
    Returns list of (state_seq, q_values) tuples for meta-learning.
    """
    X = []

    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i+seq_len].values
        next_state = df.iloc[i+seq_len]

        # Compute Q-value for each possible action
        q_vector = []
        for action_idx in range(n_actions):
            # Map discrete action to continuous [-1.0, 1.0]
            action_cont = -1.0 + 2.0 * action_idx / (n_actions - 1)
            # Compute reward for this action
            reward = compute_reward_with_action(next_state, action_cont)
            q_vector.append(reward)

        X.append((
            torch.tensor(seq, dtype=torch.float32).unsqueeze(0),
            torch.tensor(q_vector, dtype=torch.float32).unsqueeze(0)
        ))
    return X

print("\nCreating action-aware training data...")
task_data = [create_action_aware_sequences(df, SEQ_LEN, OUTPUT_DIM) for df in tasks]
print(f"Generated {len(task_data)} task datasets with action-aware Q-values")

# Initialize base model
meta_agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Meta-training loop
for episode in range(META_EPISODES):
    task_idx = random.randint(0, len(task_data) - 1)
    task_batch = task_data[task_idx]

    agent = meta_agent.clone()
    original_weights = agent.get_weights()

    # Inner training
    total_loss = 0.0
    for step in range(INNER_STEPS):
        state_seq, target_q = random.choice(task_batch)
        loss = agent.train_on_batch(state_seq, target_q.unsqueeze(0))
        total_loss += loss

    # Reptile meta-update
    updated_weights = agent.get_weights()
    new_weights = {}
    total_diff = 0.0

    for k in original_weights:
        diff = updated_weights[k] - original_weights[k]
        total_diff += torch.norm(diff).item()
        new_weights[k] = original_weights[k] + META_LR * diff

    meta_agent.set_weights(new_weights)
    print(f"[Episode {episode+1}/{META_EPISODES}] Loss: {total_loss / INNER_STEPS:.4f} | Weight Δ: {total_diff:.6f}")

# Save final meta-trained model
torch.save(meta_agent.model.state_dict(), '../models/meta_rl_model.pt')
print("\nSaved meta-trained model to meta_rl_model.pt")

# Print model stats for verification
for name, param in meta_agent.model.named_parameters():
    print(f"{name} | mean: {param.data.mean().item():.5f} | std: {param.data.std().item():.5f}")

print("Meta-training complete.")

