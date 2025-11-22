#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 29 15:15:43 2025

@author: ivanovsi
"""

# ----------------------------------------
#  LSTM-Q-Agent (Meta-RL)
# ----------------------------------------


import torch
import torch.nn as nn
import torch.optim as optim


class LSTMQNetwork(nn.Module):
    """
    Recurrent Q-network using an LSTM to encode temporal dependencies
    in building state trajectories. LSTM processes sequences of
    observations and outputs a Q-value vector for the last timestep.
    """
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0, use_layer_norm=False):
        super(LSTMQNetwork, self).__init__()
        # Recurrent encoder of state sequences
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.q_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # Process entire sequence 
        lstm_out, _ = self.lstm(x)
        # use last timestep's hidden state
        last_output = lstm_out[:, -1, :]
        # Optional stabilization 
        if self.layer_norm:
            last_output = self.layer_norm(last_output)
        # Predict Q-values for all actions
        q_values = self.q_layer(last_output)
        return q_values


class LSTMQAgent:
    """
    LSTM-based Q-learning agent that learns from experience and can be cloned/adapted
    during meta-training.
    """
    def __init__(self, input_dim, hidden_dim, output_dim,
                 num_layers=1, dropout=0.0, lr=1e-3, use_layer_norm=False):
        self.model = LSTMQNetwork(
            input_dim, hidden_dim, output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

        # For cloning
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'lr': lr,
            'use_layer_norm': use_layer_norm
        }

    def predict(self, state_seq_tensor):
        """Compute Q-values for the last timestep of a state sequence."""
        self.model.eval()
        with torch.no_grad():
            return self.model(state_seq_tensor)

    def train_on_batch(self, state_batch, target_q_values):
        """One training step using TD targets."""
        self.model.train()
        predicted_q = self.model(state_batch)
        loss = self.criterion(predicted_q, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_weights(self):
        """Return network weights (for meta-learning)."""
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        """Load network weights."""
        self.model.load_state_dict(weights)

    def clone(self):
        """Create a full copy of the agent for inner-loop adaptation."""
        new_agent = LSTMQAgent(**self.config)
        new_agent.set_weights(self.get_weights())
        return new_agent

