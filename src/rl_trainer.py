#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 11:01:30 2025

@author: ivanovsi
"""

# train_lstm_q_rl.py

import os
import random
import numpy as np
import torch
import pandas as pd

from rl_lstm_q_agent import LSTMQRLAgent
from agent_setup import load_all_building_data
from reward_utils import compute_reward_with_action

# ---------------- Config ----------------

DATA_ROOT = "../outputs/phase3"
MODEL_PATH = "../models/rl_lstm_q_model.pt"
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

INPUT_DIM = 17
HIDDEN_DIM = 64
OUTPUT_DIM = 5          # 5 discrete actions
SEQ_LEN = 8
EPOCHS = 20             # you can tune this
GAMMA = 0.99
LR = 1e-3

# Epsilon-greedy exploration
EPSILON_START = 0.2
EPSILON_END = 0.01
EPSILON_DECAY = 0.995

SEED = 2718
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

print(f"Training TRUE RL LSTM-Q agent on buildings 1–4 from: {DATA_ROOT}")


# ---------- Helper: map discrete action to continuous ----------

def map_action_to_continuous(action_idx, n_actions=OUTPUT_DIM):
    """
    Map discrete action index (0..4) to continuous control in [-1.0, 1.0].
    """
    return -1.0 + 2.0 * action_idx / (n_actions - 1)


# ---------- Helper: build sequences for (s_t, s_{t+1}) ----------

def get_state_sequence(df, start_idx, seq_len):
    """
    Return an (seq_len, INPUT_DIM) numpy array representing
    s_{t-seq_len+1},...,s_t from the dataframe.
    """
    seq = df.iloc[start_idx:start_idx + seq_len].values
    return seq  # shape: (seq_len, INPUT_DIM)


# ---------- Initialize agent and data ----------

agent = LSTMQRLAgent(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    output_dim=OUTPUT_DIM,
    lr=LR,
    gamma=GAMMA
)

all_tasks = load_all_building_data(DATA_ROOT)
train_tasks = all_tasks[:4]   # Buildings 1–4
print(f"Using {len(train_tasks)} training tasks (Building_1 to Building_4).")


# ---------- RL training loop ----------

epsilon = EPSILON_START

print("\nStarting RL training with Bellman updates...\n")
for epoch in range(EPOCHS):
    epoch_loss = 0.0
    num_updates = 0

    for task_idx, df in enumerate(train_tasks):
        # We simulate a 1-step environment over the offline time series.
        # Each "time step" we:
        #   - build a sequence S_t from df[i : i+SEQ_LEN]
        #   - choose action with epsilon-greedy
        #   - compute reward r(s_{t+1}, a)
        #   - build next sequence S_{t+1} from df[i+1 : i+1+SEQ_LEN]
        #   - perform one Q-learning update

        # we need at least SEQ_LEN+1 rows to form (S_t, S_t+1)
        max_start = len(df) - (SEQ_LEN + 1)
        if max_start <= 0:
            continue

        for i in range(max_start):
            # ----- build current and next sequences -----
            s_seq = get_state_sequence(df, i, SEQ_LEN)           # S_t
            s_next_seq = get_state_sequence(df, i + 1, SEQ_LEN)  # S_{t+1}

            # Tensors with batch dimension 1
            state_tensor = torch.tensor(s_seq, dtype=torch.float32).unsqueeze(0)
            next_state_tensor = torch.tensor(s_next_seq, dtype=torch.float32).unsqueeze(0)

            # ----- select action via epsilon-greedy -----
            action = agent.select_action(state_tensor, epsilon)
            action_cont = map_action_to_continuous(action, OUTPUT_DIM)

            # row that corresponds to "next state" for reward
            next_row = df.iloc[i + SEQ_LEN]  # same as in your supervised setup

            # ----- compute reward -----
            reward_val = compute_reward_with_action(next_row, action_cont)

            # done flag: we can mark last transition as terminal
            done = 1.0 if (i == max_start - 1) else 0.0

            # ----- prepare batch for train_step (B=1) -----
            state_batch = state_tensor                  # (1, T, D)
            next_state_batch = next_state_tensor        # (1, T, D)
            action_batch = torch.tensor([action], dtype=torch.long)
            reward_batch = torch.tensor([reward_val], dtype=torch.float32)
            done_batch = torch.tensor([done], dtype=torch.float32)

            loss = agent.train_step(
                state_batch=state_batch,
                action_batch=action_batch,
                reward_batch=reward_batch,
                next_state_batch=next_state_batch,
                done_batch=done_batch
            )
            epoch_loss += loss
            num_updates += 1

    # Epsilon decay after each epoch
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    avg_loss = epoch_loss / max(1, num_updates)
    print(f"Epoch {epoch+1}/{EPOCHS} | Updates: {num_updates} | "
          f"Avg Loss: {avg_loss:.4f} | Epsilon: {epsilon:.4f}")

# ---------- Save model ----------

torch.save(agent.model.state_dict(), MODEL_PATH)
print(f"\nRL LSTM-Q model saved to {MODEL_PATH}")
