#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 21:17:12 2025

@author: ivanovsi
"""

"""
Experience Replay Buffer for Deep Q-Learning

Stores transitions (state, action, reward, next_state, done) and enables
random sampling to break temporal correlations in training data.
"""

import numpy as np
import random
from collections import deque


class ReplayBuffer:
    """
    Circular buffer for storing and sampling experience tuples.
    
    Each experience is a tuple: (state_seq, action, reward, next_state_seq, done)
    - state_seq: numpy array of shape (seq_len, input_dim)
    - action: int (discrete action index)
    - reward: float
    - next_state_seq: numpy array of shape (seq_len, input_dim)
    - done: bool (whether episode terminated)
    """
    
    def __init__(self, capacity=10000):
        """
        Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)
        self.capacity = capacity
    
    def add(self, state_seq, action, reward, next_state_seq, done):
        """
        Add a transition to the buffer.
        
        Args:
            state_seq: State sequence (seq_len, input_dim)
            action: Action taken (int)
            reward: Reward received (float)
            next_state_seq: Next state sequence (seq_len, input_dim)
            done: Whether episode ended (bool)
        """
        self.buffer.append((state_seq, action, reward, next_state_seq, done))
    
    def sample(self, batch_size):
        """
        Sample a random batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            
        Returns:
            Tuple of numpy arrays: (states, actions, rewards, next_states, dones)
        """
        # Sample random indices
        batch = random.sample(self.buffer, batch_size)
        
        # Unpack into separate arrays
        states = np.array([t[0] for t in batch], dtype=np.float32)
        actions = np.array([t[1] for t in batch], dtype=np.int64)
        rewards = np.array([t[2] for t in batch], dtype=np.float32)
        next_states = np.array([t[3] for t in batch], dtype=np.float32)
        dones = np.array([t[4] for t in batch], dtype=np.float32)
        
        return states, actions, rewards, next_states, dones
    
    def __len__(self):
        """Return current size of buffer."""
        return len(self.buffer)
    
    def clear(self):
        """Clear all experiences from buffer."""
        self.buffer.clear()
    
    def is_ready(self, batch_size):
        """
        Check if buffer has enough samples for training.
        
        Args:
            batch_size: Minimum number of samples needed
            
        Returns:
            bool: True if buffer size >= batch_size
        """
        return len(self.buffer) >= batch_size
