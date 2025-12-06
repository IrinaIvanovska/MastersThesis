#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os


try:
    from citylearn.citylearn import CityLearnEnv
    CITYLEARN_AVAILABLE = True
except ImportError:
    CITYLEARN_AVAILABLE = False
    CityLearnEnv = None


class SimpleBuildingEnv:
    """
    Simplified building environment using pre-loaded data (fallback option).
    
    Uses static building data and simulates building dynamics.
    """
    
    def __init__(self, building_df, seq_len=8, n_actions=5, scaler=None, columns=None):
        """
        Initialize simple environment from DataFrame.
        
        Args:
            building_df: DataFrame with building data (NORMALIZED)
            seq_len: Sequence length for LSTM
            n_actions: Number of discrete actions
            scaler: MinMaxScaler for denormalization (to get absolute kW)
            columns: Column names for denormalization
        """
        self.df = building_df.reset_index(drop=True)
        self.seq_len = seq_len
        self.n_actions = n_actions
        self.current_idx = 0
        self.max_steps = len(self.df) - seq_len - 1
        
        # Store scaler for denormalization in reward
        self.scaler = scaler
        self.columns = columns
        
        # State dimension is number of features
        self.state_dim = len(self.df.columns)
    
    def reset(self):
        """Reset to beginning of data."""
        self.current_idx = 0
        
        # Return initial sequence
        seq = self.df.iloc[0:self.seq_len].values
        return seq.astype(np.float32)
    
    def step(self, action):
        """
        Simulate one step.
        
        Args:
            action: Discrete action index OR Continuous action (if continuous=True)
            
        Returns:
            tuple: (next_state_seq, reward, done, info)
        """
        # Handle Continuous vs Discrete Actions
        if isinstance(action, (float, np.floating)) or (isinstance(action, np.ndarray) and action.dtype.kind == 'f'):
            # Continuous Action: Clip to [-1, 1]
            action_cont = np.clip(float(action), -1.0, 1.0)
        else:
            # Discrete Action: Map to [-1, 1]
            action_cont = -1.0 + 2.0 * action / (self.n_actions - 1) if self.n_actions > 1 else 0.0
        
        # Get current and next states
        self.current_idx += 1
        next_state = self.df.iloc[self.current_idx + self.seq_len]
        
        # Compute reward (aligned with reward_utils.py)
        reward = self._compute_reward(next_state, action_cont)
        
        # Get next state sequence
        next_seq_start = self.current_idx
        next_seq_end = self.current_idx + self.seq_len
        next_state_seq = self.df.iloc[next_seq_start:next_seq_end].values.astype(np.float32)
        
        # Check if done
        done = (self.current_idx >= self.max_steps - 1)
        
        info = {}
        
        return next_state_seq, reward, done, info
    
    def _compute_reward(self, state, action_cont):
        """
        Compute reward (matches reward_utils.py).
        
        Args:
            state: State row (Series or dict-like) - NORMALIZED [0,1]
            action_cont: Continuous action in [-1.0, 1.0]
            
        Returns:
            float: Reward value
        """
        # Denormalize demands to absolute kW
        if self.scaler is not None and self.columns is not None:
            # Convert state Series to numpy array in correct column order
            import numpy as np
            import pandas as pd
            
            state_array = np.array([state[col] for col in self.columns]).reshape(1, -1)
            state_absolute = self.scaler.inverse_transform(state_array)[0]
            
            # Get absolute kW values
            cooling_demand_kw = state_absolute[self.columns.index('cooling_demand')]
            heating_demand_kw = state_absolute[self.columns.index('heating_demand')]
            setpoint_diff_kw = state_absolute[self.columns.index('average_unmet_cooling_setpoint_difference')]
        else:
            # Fallback to normalized
            cooling_demand_kw = float(state['cooling_demand'])
            heating_demand_kw = float(state['heating_demand'])
            setpoint_diff_kw = float(state['average_unmet_cooling_setpoint_difference'])
        
        # Simulate action effect on ABSOLUTE demands (kW)
        cooling_adj = cooling_demand_kw * (1 + 0.1 * action_cont)
        heating_adj = heating_demand_kw * (1 - 0.1 * action_cont)
        
        # Reward based on absolute kW values
        # Now a 300kW peak has 10x penalty of a 30kW peak!
        reward = -(
            0.5 * abs(setpoint_diff_kw) +
            0.25 * cooling_adj +
            0.25 * heating_adj
        )
        
        # Small penalty for large actions
        action_penalty = 0.01 * (action_cont ** 2)
        
        return reward - action_penalty
    
    def get_state_dim(self):
        """Return state dimension."""
        return self.state_dim
    
    def get_action_dim(self):
        """Return number of actions."""
        return self.n_actions
