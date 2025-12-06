#!/usr/bin/env python3
# -*- coding: utf-8 -*-


# Training RL using SAC 

import torch
import numpy as np
import random
import os
from sac_agent import SACAgent, ReplayBuffer
from agent_setup import load_all_building_data
from environment_wrapper import SimpleBuildingEnv

DATA_ROOT = '../outputs/phase3'
MODEL_PATH = '../models/sac_model.pt'
INPUT_DIM = 17 * 8  # Flattened sequence (17 features * 8 steps)
ACTION_DIM = 1      # Continuous action [-1, 1]
HIDDEN_DIM = 512
SEQ_LEN = 8

GAMMA = 0.99
TAU = 0.005
ALPHA = 0.2
LR = 0.0003
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 256
NUM_EPISODES = 200
MAX_STEPS_PER_EPISODE = 100
UPDATES_PER_STEP = 1

SEED = 2718
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

class ActionSpace:
    def __init__(self, low, high, shape):
        self.low = low
        self.high = high
        self.shape = shape

# Initialize

print("STANDARD RL WITH SAC (SOFT ACTOR-CRITIC)")
print("=" * 80)

# Load Data
all_tasks, scaler, columns = load_all_building_data(DATA_ROOT)
train_buildings = all_tasks[:4]
print(f"[+] Loaded {len(train_buildings)} training buildings")

# Create Environments
# SAC uses flattened state input 
train_envs = [SimpleBuildingEnv(df, seq_len=SEQ_LEN, n_actions=5, scaler=scaler, columns=columns) 
              for df in train_buildings]
print(f"[+] Created {len(train_envs)} environments")

# Initialize Agent
action_space = ActionSpace(np.array([-1.0]), np.array([1.0]), (1,))
agent = SACAgent(INPUT_DIM, action_space, hidden_dim=HIDDEN_DIM, gamma=GAMMA, tau=TAU, alpha=ALPHA, lr=LR)
memory = ReplayBuffer(REPLAY_BUFFER_SIZE, (INPUT_DIM,), 1)
print("SAC Agent and Replay Buffer initialized")

# Training Loop

print("\n Starting Training...")

total_steps = 0
episode_rewards = []

for episode in range(NUM_EPISODES):
    # Select random building
    env = random.choice(train_envs)
    
    # Reset
    state_seq = env.reset()
    state_flat = state_seq.flatten() # Flatten sequence for SAC
    episode_reward = 0
    
    for step in range(MAX_STEPS_PER_EPISODE):
        # Select Action
        if len(memory) > BATCH_SIZE:
            action = agent.select_action(state_flat)
        else:
            action = np.random.uniform(-1, 1, (1,)) # Random exploration start
            
        # Step
        next_state_seq, reward, done, info = env.step(action[0]) # Pass float action
        next_state_flat = next_state_seq.flatten()
        
        # Store
        mask = 1 if step == MAX_STEPS_PER_EPISODE - 1 else float(not done)
        memory.add(state_flat, action, reward, next_state_flat, mask)
        
        # Train
        if len(memory) > BATCH_SIZE:
            for i in range(UPDATES_PER_STEP):
                agent.update_parameters(memory, BATCH_SIZE, total_steps)
            
        # Update State
        state_flat = next_state_flat
        episode_reward += reward
        total_steps += 1
        
        if done:
            break
            
    episode_rewards.append(episode_reward)
    
    if (episode + 1) % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode+1:3d}/{NUM_EPISODES} | Reward: {episode_reward:7.2f} | Avg10: {avg_reward:7.2f}")

# Save
agent.save_model(MODEL_PATH)
print(f"\n SAC Model saved to {MODEL_PATH}")
