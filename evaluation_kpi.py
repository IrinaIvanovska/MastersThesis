#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 20:30:58 2025

@author: ivanovsi
"""

# Modified to compute and store KPI metrics after evaluation

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data

# Config
DATA_ROOT = 'outputs/phase3'
OUTPUT_DIR = 'outputs/both'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INPUT_DIM = 27
HIDDEN_DIM = 64
OUTPUT_DIM = 5
SEQ_LEN = 8
BUILDINGS = [f'Building_{i+1}' for i in range(6)]
EPSILON = 0.2
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)

def create_sequences(df, seq_len=8):
    X = []
    for i in range(len(df) - seq_len):
        seq = df.iloc[i:i+seq_len].values
        X.append(torch.tensor(seq, dtype=torch.float32).unsqueeze(0))
    return X

def reward_proxy(row, action):
    base_reward = -(0.7 * abs(float(row['average_unmet_cooling_setpoint_difference'])) +
                    0.3 * float(row['cooling_demand']) +
                    0.3 * float(row['heating_demand']) +
                    0.1 * abs(float(row['solar_generation'])) +
                    0.05 * float(row['non_shiftable_load']))
    action_penalty = 0.1 * (action ** 2)
    demand_adjustment = 0.2 * action * float(row['cooling_demand'])
    setpoint_adjustment = 0.1 * action * abs(float(row['average_unmet_cooling_setpoint_difference']))
    return base_reward - action_penalty - demand_adjustment + setpoint_adjustment

def map_action_to_continuous(action_idx):
    return -1.0 + 2.0 * action_idx / (OUTPUT_DIM - 1)

def compute_kpis_from_obs(df):
    demand = (df["non_shiftable_load"] + df["dhw_demand"] + df["cooling_demand"] + df["heating_demand"] - df["solar_generation"]).clip(lower=0.0)
    ramping = np.mean(np.abs(np.diff(demand)))
    avg_power = np.mean(demand)
    peak_power = np.max(demand)
    load_factor = avg_power / peak_power if peak_power > 0 else 0
    one_minus_load_factor = 1 - load_factor

    if not np.issubdtype(df.index.dtype, np.datetime64):
        df = df.copy()
        df["datetime"] = pd.date_range(start="2025-01-01", periods=len(df), freq="H")
        df.set_index("datetime", inplace=True)

    daily_peak = df.resample("D")["non_shiftable_load"].max()
    average_peak_demand = daily_peak.mean()
    annual_peak_demand = peak_power
    net_electricity = demand.sum()

    return {
        "Ramping": ramping,
        "1-Load-Factor": one_minus_load_factor,
        "Average Peak Demand": average_peak_demand,
        "Annual Peak Demand": annual_peak_demand,
        "Net Electricity Consumption": net_electricity
    }

def evaluate_agent(agent_weights_path, label):
    print(f"\n📥 Evaluating {label}...")
    agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM)
    agent.model.load_state_dict(torch.load(agent_weights_path))
    agent.model.eval()

    task_data = load_all_building_data(DATA_ROOT)
    results = []

    for i, df in enumerate(task_data):
        building_name = BUILDINGS[i]
        sequences = create_sequences(df, SEQ_LEN)

        actions, q_values, continuous_actions, rewards = [], [], [], []

        for j, seq in enumerate(sequences):
            q_vals = agent.predict(seq).squeeze().numpy()
            action = np.random.randint(0, OUTPUT_DIM) if np.random.random() < EPSILON else np.argmax(q_vals)
            cont_action = map_action_to_continuous(action)
            reward = reward_proxy(df.iloc[j+SEQ_LEN], cont_action)

            actions.append(action)
            q_values.append(q_vals)
            continuous_actions.append(cont_action)
            rewards.append(reward)

        avg_reward = np.mean(rewards) if rewards else 0.0
        q_std = np.std(q_values) if q_values else 0.0

        kpis = compute_kpis_from_obs(df)

        results.append({
            'building': building_name,
            'avg_reward': avg_reward,
            'actions': actions,
            'q_values': q_values,
            'q_std': q_std,
            'continuous_actions': continuous_actions,
            'rewards': rewards,
            'kpis': kpis
        })

        pd.DataFrame(kpis, index=[0]).to_csv(os.path.join(OUTPUT_DIR, f'{building_name}_{label}_kpis.csv'), index=False)
        print(f"🏢 {building_name} - Avg Reward ({label}): {avg_reward:.3f}, Q-Value Std: {q_std:.3f}")
        for k, v in kpis.items():
            print(f"    {k}: {v:.3f}")

    return results