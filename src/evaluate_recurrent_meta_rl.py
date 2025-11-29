#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 28 20:36:24 2025

@author: ivanovsi
"""



import torch
import numpy as np
import pandas as pd
import os
from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data
from environment_wrapper import SimpleBuildingEnv
from recurrent_env_wrapper import RecurrentBuildingEnv

OUTPUT_DIR = '../outputs/recurrent_results'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 23
HIDDEN_DIM = 128
OUTPUT_DIM = 5
SEQ_LEN = 8
TEST_BUILDINGS = ['Building_5', 'Building_6']

def map_action_to_continuous(action_idx):
    return -1.0 + 2.0 * action_idx / (OUTPUT_DIM - 1)

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

def evaluate_recurrent_agent(model_path, label):
    print(f"\n {label}...")
    
    agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    agent.model.load_state_dict(torch.load(model_path))
    agent.model.eval()
    
    all_tasks, scaler, columns = load_all_building_data('../outputs/phase3')
    test_tasks = all_tasks[4:6]
    
    results = []
    
    for i, df in enumerate(test_tasks):
        building_name = TEST_BUILDINGS[i]
        
        # Create Environment
        base_env = SimpleBuildingEnv(df, seq_len=SEQ_LEN, n_actions=OUTPUT_DIM, scaler=scaler, columns=columns)
        env = RecurrentBuildingEnv(base_env)
        
        state_seq = env.reset()
        actions_continuous = []
        
        done = False
        while not done:
            # Predict
            state_tensor = torch.tensor(state_seq, dtype=torch.float32).unsqueeze(0)
            q_vals = agent.predict(state_tensor).squeeze().detach().numpy()
            action = np.argmax(q_vals)
            
            # Step
            state_seq, _, done, _ = env.step(action)
            
            # Store
            cont_action = map_action_to_continuous(action)
            actions_continuous.append(cont_action)
            
        metrics = calculate_metrics(df, actions_continuous, label, building_name)
        results.append(metrics)
        print(f"  {building_name}: Cost {metrics['cost_savings_pct']:+7.2f}%")
        
    return results

print("="*80)
print("EVALUATING RECURRENT META-RL")
print("="*80)

results = evaluate_recurrent_agent('../models/meta_rl_recurrent.pt', 'Recurrent Meta-RL (RL^2)')

avg_cost = np.mean([r['cost_savings_pct'] for r in results])
print(f"\n[+] Average Cost Savings: {avg_cost:+.2f}%")