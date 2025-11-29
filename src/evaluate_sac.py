#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 20:46:25 2025

@author: ivanovsi
"""

import torch
import numpy as np
import pandas as pd
import os
from sac_agent import SACAgent
from agent_setup import load_all_building_data
from environment_wrapper import SimpleBuildingEnv

OUTPUT_DIR = '../outputs/sac_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 17 * 8
HIDDEN_DIM = 256
SEQ_LEN = 8
TEST_BUILDINGS = ['Building_5', 'Building_6']

class ActionSpace:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

def calculate_metrics(df, actions_continuous, label, building_name):
    baseline_cooling = df['cooling_demand'].values
    baseline_heating = df['heating_demand'].values
    baseline_non_shiftable = df['non_shiftable_load'].values
    baseline_consumption = baseline_cooling + baseline_heating + baseline_non_shiftable
    baseline_cost = (df['electricity_pricing'] * baseline_consumption).sum()
    
    action_series = pd.Series(actions_continuous + [0.0] * SEQ_LEN)[:len(df)]
    
    controlled_cooling = df['cooling_demand'] * (1 + 0.1 * action_series)
    controlled_heating = df['heating_demand'] * (1 - 0.1 * action_series)
    controlled_consumption = (controlled_cooling + controlled_heating + df['non_shiftable_load']).values
    controlled_cost = (df['electricity_pricing'] * pd.Series(controlled_consumption)).sum()
    
    cost_savings = ((baseline_cost - controlled_cost) / baseline_cost * 100)
    
    return {'building': building_name, 'label': label, 'cost_savings_pct': cost_savings}

def evaluate_sac_agent(model_path, label):
    print(f"\n {label}...")
    
    action_space = ActionSpace(np.array([-1.0]), np.array([1.0]), (1,))
    agent = SACAgent(INPUT_DIM, action_space, hidden_dim=HIDDEN_DIM)
    agent.load_model(model_path)
    
    all_tasks, scaler, columns = load_all_building_data('../outputs/phase3')
    test_tasks = all_tasks[4:6]
    
    results = []
    
    for i, df in enumerate(test_tasks):
        building_name = TEST_BUILDINGS[i]
        
        env = SimpleBuildingEnv(df, seq_len=SEQ_LEN, n_actions=5, scaler=scaler, columns=columns)
        
        state_seq = env.reset()
        state_flat = state_seq.flatten()
        actions_continuous = []
        
        done = False
        while not done:
            # Predict
            action = agent.select_action(state_flat, evaluate=True)
            
            # Step
            next_state_seq, _, done, _ = env.step(action[0])
            next_state_flat = next_state_seq.flatten()
            
            # Store
            actions_continuous.append(action[0])
            
            state_flat = next_state_flat
            
        metrics = calculate_metrics(df, actions_continuous, label, building_name)
        results.append(metrics)
        print(f"  {building_name}: Cost {metrics['cost_savings_pct']:+7.2f}%")
        
    return results

print("="*80)
print("EVALUATING STANDARD RL (SAC)")
print("="*80)

results = evaluate_sac_agent('../models/sac_model.pt', 'Standard RL (SAC)')

avg_cost = np.mean([r['cost_savings_pct'] for r in results])
print(f"\n[+] Average Cost Savings: {avg_cost:+.2f}%")
