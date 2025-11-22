#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 13:34:21 2025

@author: ivanovsi
"""

# ----------------------------------------
# Train Standard LSTM-Q Agent (No Meta-RL)
# ----------------------------------------

import torch
from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data
from reward_utils import compute_reward_with_action
import numpy as np
import random

# Config
DATA_ROOT = '../outputs/phase3'
MODEL_PATH = '../models/rl_model.pt'
INPUT_DIM = 17
HIDDEN_DIM = 64
OUTPUT_DIM = 5  # Arbitrary action size
SEQ_LEN = 8
EPOCHS = 10
SEED = 2718  # Set random seed for reproducibility

# Set all random seeds
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Create action-aware training sequences
def create_action_aware_sequences(df, seq_len=8, n_actions=5):
    """
    Create training data with action-dependent rewards.
    For each sequence, compute Q-values for all possible actions.
    """
    X = []
    Y = []

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

        X.append(seq)
        Y.append(q_vector)

    return torch.tensor(X, dtype=torch.float32), torch.tensor(Y, dtype=torch.float32)

# Initialize agent
agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)

# Load data for training buildings only (1-4)
all_tasks = load_all_building_data(DATA_ROOT)
tasks = all_tasks[:4]  # Use only first 4 buildings for training
print(f"Training on {len(tasks)} buildings (Building_1 to Building_4)")

# Training loop with action-aware Q-learning
print("\nTraining with action-aware rewards...")
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    for df in tasks:
        X, Y = create_action_aware_sequences(df, SEQ_LEN, OUTPUT_DIM)
        loss = agent.train_on_batch(X, Y)
        epoch_loss += loss
    print(f"Epoch {epoch+1}/{EPOCHS} - Avg Loss: {epoch_loss / len(tasks):.4f}")

# Save model
torch.save(agent.model.state_dict(), MODEL_PATH)
print(f"\nStandard RL model saved as {MODEL_PATH}")

