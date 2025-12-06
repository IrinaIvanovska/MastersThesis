#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


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
                 num_layers=1, dropout=0.0, lr=1e-3, use_layer_norm=False, gamma=0.99):
        # Main Q-network
        self.model = LSTMQNetwork(
            input_dim, hidden_dim, output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
        
        # Target Q-network for stable Bellman updates
        self.target_model = LSTMQNetwork(
            input_dim, hidden_dim, output_dim,
            num_layers=num_layers,
            dropout=dropout,
            use_layer_norm=use_layer_norm
        )
        # Initialize target network with same weights
        self.update_target_network()
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.gamma = gamma  # Discount factor for Bellman equation
        self.output_dim = output_dim

        # For cloning
        self.config = {
            'input_dim': input_dim,
            'hidden_dim': hidden_dim,
            'output_dim': output_dim,
            'num_layers': num_layers,
            'dropout': dropout,
            'lr': lr,
            'use_layer_norm': use_layer_norm,
            'gamma': gamma
        }

    def predict(self, state_seq_tensor):
        self.model.eval()
        with torch.no_grad():
            return self.model(state_seq_tensor)

    def train_on_batch(self, state_batch, target_q_values):
        """Legacy supervised training (for backward compatibility)."""
        self.model.train()
        predicted_q = self.model(state_batch)
        loss = self.criterion(predicted_q, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def train_on_batch_bellman(self, states, actions, rewards, next_states, dones):
        """
        Train using Bellman equation (TRUE RL).
        
        Implements: Q(s,a) = r + γ * max_a' Q_target(s', a')
        
        Args:
            states: Batch of state sequences (batch_size, seq_len, input_dim)
            actions: Batch of actions taken (batch_size,)
            rewards: Batch of rewards (batch_size,)
            next_states: Batch of next state sequences (batch_size, seq_len, input_dim)
            dones: Batch of done flags (batch_size,)
            
        Returns:
            float: TD loss value
        """
        self.model.train()
        
        # Convert to tensors if needed
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
        
        # Get current Q-values for selected actions
        current_q_values = self.model(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q-values using Bellman equation
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q = next_q_values.max(dim=1)[0]
            # Bellman equation: Q(s,a) = r + γ * max_a' Q(s', a') * (1 - done)
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute TD loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def train_on_batch_double_dqn(self, states, actions, rewards, next_states, dones):
        """
        Train using Double DQN (reduces overestimation bias).
        
        Double DQN: Use main network to SELECT action, target network to EVALUATE it.
        Standard: target_q = r + γ * max Q_target(s', a')
        Double:   target_q = r + γ * Q_target(s', argmax Q_main(s', a'))
        
        Args:
            states: Batch of state sequences
            actions: Batch of actions taken
            rewards: Batch of rewards
            next_states: Batch of next state sequences
            dones: Batch of done flags
            
        Returns:
            float: TD loss value
        """
        self.model.train()
        
        # Convert to tensors if needed
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.int64)
            rewards = torch.tensor(rewards, dtype=torch.float32)
            next_states = torch.tensor(next_states, dtype=torch.float32)
            dones = torch.tensor(dones, dtype=torch.float32)
        
        # Get current Q-values for selected actions
        current_q_values = self.model(states)
        current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Double DQN: Use main network to select actions, target network to evaluate
        with torch.no_grad():
            # Main network selects actions
            next_q_values_main = self.model(next_states)
            best_actions = next_q_values_main.argmax(dim=1)
            
            # Target network evaluates those actions
            next_q_values_target = self.target_model(next_states)
            max_next_q = next_q_values_target.gather(1, best_actions.unsqueeze(1)).squeeze(1)
            
            # Double DQN Bellman target
            target_q = rewards + self.gamma * max_next_q * (1 - dones)
        
        # Compute TD loss
        loss = self.criterion(current_q, target_q)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def select_action(self, state_seq, epsilon=0.0):
        """
        Epsilon-greedy action selection.
        
        Args:
            state_seq: State sequence (seq_len, input_dim) or (1, seq_len, input_dim)
            epsilon: Exploration probability
            
        Returns:
            int: Selected action index
        """
        if np.random.random() < epsilon:
            # Random exploration
            return np.random.randint(0, self.output_dim)
        else:
            # Greedy exploitation
            self.model.eval()
            with torch.no_grad():
                # Add batch dimension if needed
                if len(state_seq.shape) == 2:
                    state_seq = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0)
                elif not isinstance(state_seq, torch.Tensor):
                    state_seq = torch.tensor(state_seq, dtype=torch.float32)
                    
                q_values = self.model(state_seq)
                return q_values.argmax(dim=1).item()
    
    def update_target_network(self):
        """
        Copy weights from main network to target network.
        Should be called periodically during training.
        """
        self.target_model.load_state_dict(self.model.state_dict())

    def get_weights(self):
        return {k: v.clone() for k, v in self.model.state_dict().items()}

    def set_weights(self, weights):
        self.model.load_state_dict(weights)

    def clone(self):
        new_agent = LSTMQAgent(**self.config)
        new_agent.set_weights(self.get_weights())
        return new_agent

