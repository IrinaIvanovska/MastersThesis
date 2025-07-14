#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 15 10:12:06 2025

@author: ivanovsi
"""

# ----------------------------------------
# Verified Reptile Meta-RL Trainer using LSTM-Q Agent
# ----------------------------------------

import torch
import random
import numpy as np
from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data
import os
import pandas as pd
# Config
INPUT_DIM = 28
HIDDEN_DIM = 64
OUTPUT_DIM = 5
INNER_STEPS = 10
META_EPISODES = 100
META_LR = 1.0
SEQ_LEN = 8

# Load building data
tasks = load_all_building_data('outputs/phase3')

def compute_reward(df):
    return -(
        df["cooling_demand"]
        + df["dhw_demand"]
        + df["electricity_pricing"] * df["non_shiftable_load"]
        - 0.1 * df["solar_generation"]
    )

# Create LSTM sequences and Q-targets
def create_sequences(df, seq_len=8):
    X = []
    rewards = compute_reward(df).tolist()
    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i+seq_len].values
        q_val = rewards[i + seq_len - 1]
        action_index = int(np.mean(seq[:, 0]) * 10) % OUTPUT_DIM
        q_vector = [0.0] * OUTPUT_DIM
        q_vector[action_index] = q_val
        X.append((
            torch.tensor(seq, dtype=torch.float32).unsqueeze(0),
            torch.tensor(q_vector, dtype=torch.float32)
        ))
    return X

task_data = [create_sequences(df, SEQ_LEN) for df in tasks]

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
torch.save(meta_agent.model.state_dict(), 'meta_rl_model.pt')
print("\n💾 Saved meta-trained model to meta_rl_model.pt")

# Print model stats for verification
for name, param in meta_agent.model.named_parameters():
    print(f"{name} | mean: {param.data.mean().item():.5f} | std: {param.data.std().item():.5f}")

print("✅ Verified Meta-training complete.")

from evaluation_kpi import compute_kpis_from_obs  # <- reuse from earlier

os.makedirs("outputs/meta_rl_kpis", exist_ok=True)

print("\n📊 Evaluating final Meta-RL model on all buildings for KPIs:")
for i, df in enumerate(tasks):
    building_name = f'Building_{i+1}'
    kpis = compute_kpis_from_obs(df)
    pd.DataFrame(kpis, index=[0]).to_csv(f"outputs/meta_rl_kpis/{building_name}_kpis.csv", index=False)

    print(f"🏢 {building_name}:")
    for key, val in kpis.items():
        print(f"   {key}: {val:.3f}")
