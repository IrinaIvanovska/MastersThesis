#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 19 19:48:23 2025

@author: ivanovsi
"""

import torch
import torch.nn as nn
import torch.optim as optim


class LSTMQNetwork(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=1, dropout=0.0, use_layer_norm=False):
        super(LSTMQNetwork, self).__init__()
        self.lstm = nn.LSTM(
            input_dim, hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(hidden_dim) if use_layer_norm else None
        self.q_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        if self.layer_norm:
            last_output = self.layer_norm(last_output)
        q_values = self.q_layer(last_output)
        return q_values


class LSTMQAgent:
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
        self.model.eval()
        with torch.no_grad():
            return self.model(state_seq_tensor)

    def train_on_batch(self, state_batch, target_q_values):
        self.model.train()
        predicted_q = self.model(state_batch)
        loss = self.criterion(predicted_q, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def get_weights(self):
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def clone(self):
        new_agent = LSTMQAgent(**self.config)
        new_agent.set_weights(self.get_weights())
        return new_agent

