#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 19:34:37 2025

@author: ivanovsi
"""

# ----------------------------------------
# Train Standard LSTM-Q Agent (No Meta-RL)
# ----------------------------------------

import torch
import os
from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data
import numpy as np

# Config
DATA_ROOT = 'outputs/phase3'
MODEL_PATH = 'rl_model.pt'
INPUT_DIM = 28
HIDDEN_DIM = 64
OUTPUT_DIM = 5  # Arbitrary action size
SEQ_LEN = 8
EPOCHS = 10

# Reward Proxy
def compute_reward(df):
    return -(
        df["cooling_demand"]
        + df["dhw_demand"]
        + df["electricity_pricing"] * df["non_shiftable_load"]
        - 0.1 * df["solar_generation"]
    )

# Create sequences
def make_sequences(df, reward, seq_len):
    X, Y = [], []
    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i+seq_len].values
        X.append(seq)
        q_val = reward[i+seq_len-1]
        q_dummy = [0.0] * OUTPUT_DIM
        q_dummy[np.random.randint(0, OUTPUT_DIM)] = q_val
        Y.append(q_dummy)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Initialize agent
agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Load data for all buildings
tasks = load_all_building_data(DATA_ROOT)

# Training loop
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for df in tasks:
        reward = compute_reward(df).tolist()
        X, Y = make_sequences(df, reward, SEQ_LEN)
        loss = agent.train_on_batch(X, Y)
        epoch_loss += loss
    print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {epoch_loss / len(tasks):.4f}")

# Save model
torch.save(agent.model.state_dict(), MODEL_PATH)
print(f"\n✅ Standard RL model saved as {MODEL_PATH}")

