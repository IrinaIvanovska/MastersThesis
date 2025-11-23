#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 10:55:01 2025

@author: ivanovsi
"""

# lstm_q_rl_agent.py

import torch
import torch.nn as nn
import torch.optim as optim
from lstm_q_agent import LSTMQNetwork
 


class LSTMQRLAgent:
    """
    LSTM-based Q-learning agent.
    Uses an LSTM encoder + linear Q-value head + Bellman updates.
    """
    def __init__(self,
                 input_dim,
                 hidden_dim,
                 output_dim,
                 num_layers=1,
                 dropout=0.0,
                 lr=1e-3,
                 gamma=0.99,
                 use_layer_norm=False):

        self.model = LSTMQNetwork(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma
        self.output_dim = output_dim

    # ---------- Inference ----------

    def predict(self, state_seq_tensor):
        """
        Compute Q-values for the last timestep of a state sequence.
        state_seq_tensor: (1, seq_len, input_dim) or (B, seq_len, input_dim)
        """
        self.model.eval()
        with torch.no_grad():
            return self.model(state_seq_tensor)

    def select_action(self, state_seq_tensor, epsilon):
        """
        Epsilon-greedy action selection on a single sequence.
        state_seq_tensor: (1, seq_len, input_dim)
        returns: int action index
        """
        import random
        if random.random() < epsilon:
            return random.randint(0, self.output_dim - 1)
        q_values = self.predict(state_seq_tensor)  # (1, A)
        action = int(torch.argmax(q_values, dim=1).item())
        return action

    # ---------- RL training step (Bellman update) ----------

    def train_step(self, state_batch, action_batch, reward_batch,
                   next_state_batch, done_batch):
        """
        One Q-learning update using Bellman targets.

        state_batch:      (B, seq_len, input_dim)
        action_batch:     (B,)          long
        reward_batch:     (B,)          float
        next_state_batch: (B, seq_len, input_dim)
        done_batch:       (B,)          0.0 (not done) or 1.0 (done)
        """
        self.model.train()

        state_batch = state_batch
        next_state_batch = next_state_batch
        action_batch = action_batch
        reward_batch = reward_batch
        done_batch = done_batch

        # Current Q(s,a) for all actions
        q_pred = self.model(state_batch)            # (B, A)

        # Q(s', a') for all actions in next state
        with torch.no_grad():
            q_next = self.model(next_state_batch)   # (B, A)
            max_next_q, _ = q_next.max(dim=1)      # (B,)

            # Bellman target for chosen action:
            # y = r + gamma * max_a' Q(s', a')   if not done
            # y = r                              if done
            target_for_action = reward_batch + (1.0 - done_batch) * self.gamma * max_next_q

            # Start from the current predictions and replace only the chosen actions
            q_target = q_pred.clone()
            batch_indices = torch.arange(q_pred.size(0))
            q_target[batch_indices, action_batch] = target_for_action

        # MSE between predicted Q(s,·) and updated Q-target vector
        loss = self.criterion(q_pred, q_target.detach())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return float(loss.item())
