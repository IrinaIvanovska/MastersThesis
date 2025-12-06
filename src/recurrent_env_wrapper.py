#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import torch
from collections import deque

class RecurrentBuildingEnv:
    """
    Wrapper for SimpleBuildingEnv that augments the state with 
    previous action and previous reward for Recurrent Meta-RL (RL^2).
    
    Input:
        - Base State (17 dims)
        - Previous Action (One-hot, 5 dims)
        - Previous Reward (1 dim)
    Total Dim: 23
    """
    
    def __init__(self, env):
        self.env = env
        self.seq_len = env.seq_len
        self.n_actions = env.n_actions
        self.state_dim = env.state_dim + self.n_actions + 1 # 17 + 5 + 1 = 23
        
        # History buffer for sequence generation
        self.history = deque(maxlen=self.seq_len)
        
    def reset(self):
        # Reset base environment
        base_seq = self.env.reset()
        
        # Clear history
        self.history.clear()
        
        # Initialize history with first state and zero action/reward
        # base_seq is (seq_len, 17), we take the last one as "current" state
        
        for i in range(len(base_seq)):
            state = base_seq[i]
            prev_action = np.zeros(self.n_actions, dtype=np.float32)
            prev_reward = np.array([0.0], dtype=np.float32)
            
            aug_state = np.concatenate([state, prev_action, prev_reward])
            self.history.append(aug_state)
            
        return np.array(self.history, dtype=np.float32)
        
    def step(self, action_idx):
        # Execute action in base environment
        next_base_seq, reward, done, info = self.env.step(action_idx)
        
        # SimpleBuildingEnv returns (seq_len, 17). The last row is the new state.
        next_state = next_base_seq[-1]
        
        # Create One-Hot Action
        prev_action = np.zeros(self.n_actions, dtype=np.float32)
        prev_action[action_idx] = 1.0
        
        # Previous Reward
        prev_reward = np.array([reward], dtype=np.float32)
        
        # Construct Augmented State
        aug_state = np.concatenate([next_state, prev_action, prev_reward])
        
        # Update History
        self.history.append(aug_state)
        
        # Return Sequence
        return np.array(self.history, dtype=np.float32), reward, done, info
        
    def get_state_dim(self):
        return self.state_dim
    
    def get_action_dim(self):
        return self.n_actions
