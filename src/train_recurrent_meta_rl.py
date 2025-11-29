#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 06 20:28:14 2025

@author: ivanovsi
"""

# ============================================================================
# Training Script for Recurrent Meta-RL (RL^2)
# ============================================================================

import torch
import numpy as np
import random
from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data
from environment_wrapper import SimpleBuildingEnv
from recurrent_env_wrapper import RecurrentBuildingEnv
from replay_buffer import ReplayBuffer

# ============================================================================
# Configuration
# ============================================================================

DATA_ROOT = '../outputs/phase3'
MODEL_PATH = '../models/meta_rl_recurrent.pt'
INPUT_DIM = 23  # 17 (State) + 5 (Prev Action) + 1 (Prev Reward)
HIDDEN_DIM = 128 # Increased for meta-learning capacity
OUTPUT_DIM = 5
SEQ_LEN = 8

# RL Hyperparameters
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.995
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 32
TARGET_UPDATE_FREQ = 100
NUM_EPISODES = 100
MAX_STEPS_PER_EPISODE = 100

# Reproducibility
SEED = 2718
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ============================================================================
# Initialize
# ============================================================================

print("=" * 80)
print("RECURRENT META-RL (RL^2) TRAINING")
print("=" * 80)

# Load Data
# Load Data
all_tasks, scaler, columns = load_all_building_data(DATA_ROOT)
train_buildings = all_tasks[:4]
print(f"[+] Loaded {len(train_buildings)} training buildings")

# Create Recurrent Environments
train_envs = []
for df in train_buildings:
    base_env = SimpleBuildingEnv(df, seq_len=SEQ_LEN, n_actions=OUTPUT_DIM, scaler=scaler, columns=columns)
    rec_env = RecurrentBuildingEnv(base_env)
    train_envs.append(rec_env)

print(f"[+] Created {len(train_envs)} Recurrent Environments (Input Dim: {INPUT_DIM})")

# Initialize Agent
agent = LSTMQAgent(INPUT_DIM, HIDDEN_DIM, OUTPUT_DIM, gamma=GAMMA)
replay_buffer = ReplayBuffer(capacity=REPLAY_BUFFER_SIZE)
print("[+] Agent and Replay Buffer initialized")

print("\n Starting Training...")

epsilon = EPSILON_START
total_steps = 0
episode_rewards = []

for episode in range(NUM_EPISODES):
    # Sample Task (Building)
    env = random.choice(train_envs)
    
    # Reset
    state_seq = env.reset()
    episode_reward = 0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        # Select Action
        action = agent.select_action(state_seq, epsilon)
        
        # Step
        next_state_seq, reward, done, info = env.step(action)
        
        # Store
        replay_buffer.add(state_seq, action, reward, next_state_seq, done)
        
        # Train
        if replay_buffer.is_ready(BATCH_SIZE):
            states, actions, rewards, next_states, dones = replay_buffer.sample(BATCH_SIZE)
            agent.train_on_batch_bellman(states, actions, rewards, next_states, dones)
            
        # Update Target
        if total_steps % TARGET_UPDATE_FREQ == 0:
            agent.update_target_network()
            
        # Update State
        state_seq = next_state_seq
        episode_reward += reward
        total_steps += 1
        
        if done:
            break
            
    # Decay Epsilon
    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
    episode_rewards.append(episode_reward)
    
    if (episode + 1) % 10 == 0:
        print(f"Episode {episode+1:3d}/{NUM_EPISODES} | Reward: {episode_reward:7.2f} | ε: {epsilon:.3f}")

# Save
torch.save(agent.model.state_dict(), MODEL_PATH)
print(f"\n Recurrent Meta-RL Model saved to {MODEL_PATH}")
