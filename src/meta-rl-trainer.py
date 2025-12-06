#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#  Meta-RL with LSTM-Q and Reptile
#  - Inner loop: task-specific RL adaptation (Bellman updates)
#  - Outer loop: Reptile meta-update on initialization

import os
import random
import numpy as np
import torch

from lstm_q_agent import LSTMQAgent
from agent_setup import load_all_building_data
from environment_wrapper import SimpleBuildingEnv
from recurrent_env_wrapper import RecurrentBuildingEnv
  

# Config
DATA_ROOT = '../outputs/phase3'          
MODEL_PATH = '../models/meta_lstmq_reptile.pt'

SEED = 2718
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# LSTM-Q architecture
INPUT_DIM = 23          # number of features per timestep (must match env)
HIDDEN_DIM = 128
OUTPUT_DIM = 5          # number of discrete actions

SEQ_LEN = 8             # length of state sequence window

# Meta-RL (Reptile) hyperparameters
META_EPISODES = 400     # outer-loop iterations
INNER_STEPS = 200       # number of RL steps (transitions) per inner adaptation
META_LR = 1.0           # meta learning rate for Reptile

# RL hyperparameters (inner loop)
GAMMA = 0.99
EPSILON_START = 0.1     # small exploration during meta-training
EPSILON_END = 0.05
EPSILON_DECAY = 0.999

MAX_STEPS_PER_EPISODE = 500  # cap per episode within a task



def make_train_envs():
    """Create recurrent environments for training buildings (tasks)."""
    all_tasks, scaler, columns = load_all_building_data(DATA_ROOT)
    # use first 4 buildings as training tasks
    train_dfs = all_tasks[:4]

    envs = []
    for df in train_dfs:
        base_env = SimpleBuildingEnv(
            df,
            seq_len=SEQ_LEN,
            n_actions=OUTPUT_DIM,
            scaler=scaler,
            columns=columns
        )
        rec_env = RecurrentBuildingEnv(base_env)
        envs.append(rec_env)

    print(f"[+] Created {len(envs)} training tasks (buildings 1–4)")
    return envs


def one_inner_rl_adaptation(agent, env, inner_steps, epsilon):
    """
    Perform task-specific RL adaptation on a single environment (task).

    INNER LOOP:
    - run environment for 'inner_steps' transitions
    - use Bellman updates (train_on_batch_bellman) to adapt agent parameters
    """
    total_reward = 0.0
    steps_done = 0

    # reset environment & get initial sequence state
    state_seq = env.reset()  # expected shape: (seq_len, input_dim)
    for _ in range(inner_steps):
        # epsilon-greedy action selection
        action = agent.select_action(state_seq, epsilon=epsilon)

        # environment step
        next_state_seq, reward, done, info = env.step(action)

        # build batch of size 1 for Bellman update
        states = np.expand_dims(state_seq, axis=0)       # (1, seq_len, input_dim)
        next_states = np.expand_dims(next_state_seq, 0)  # (1, seq_len, input_dim)
        actions = np.array([action], dtype=np.int64)
        rewards = np.array([reward], dtype=np.float32)
        dones = np.array([float(done)], dtype=np.float32)

        # TRUE RL update (Bellman)
        _ = agent.train_on_batch_bellman(
            states=states,
            actions=actions,
            rewards=rewards,
            next_states=next_states,
            dones=dones
        )

        # move forward
        state_seq = next_state_seq
        total_reward += reward
        steps_done += 1

        if done or steps_done >= MAX_STEPS_PER_EPISODE:
            state_seq = env.reset()
            steps_done = 0  # episode counter inside this task

    return total_reward


def reptile_meta_training():
    # Build tasks (buildings)
    envs = make_train_envs()
    n_tasks = len(envs)

    # Initialize 
    meta_agent = LSTMQAgent(
        input_dim=INPUT_DIM,
        hidden_dim=HIDDEN_DIM,
        output_dim=OUTPUT_DIM,
        gamma=GAMMA
    )
    print("[+] Meta-agent (LSTM-Q) initialized")

    epsilon = EPSILON_START

    # Reptile
    for episode in range(1, META_EPISODES + 1):
        # 1) Sample a task 
        task_idx = random.randint(0, n_tasks - 1)
        env = envs[task_idx]

        # 2) Clone meta-agent -> task-specific copy
        task_agent = meta_agent.clone()
        original_weights = meta_agent.get_weights()

        # 3) Inner loop: RL adaptation on this task
        inner_return = one_inner_rl_adaptation(
            agent=task_agent,
            env=env,
            inner_steps=INNER_STEPS,
            epsilon=epsilon
        )

        # 4) Outer loop: Reptile meta-update
        updated_weights = task_agent.get_weights()
        new_weights = {}
        total_diff = 0.0

        for k in original_weights:
            diff = updated_weights[k] - original_weights[k]
            total_diff += torch.norm(diff).item()
            new_weights[k] = original_weights[k] + META_LR * diff

        meta_agent.set_weights(new_weights)

        # 5) Epsilon decay (optional, small amount of exploration)
        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

        # Logging
        avg_inner_return = inner_return / max(1, INNER_STEPS)
        print(
            f"[MetaEp {episode:03d}/{META_EPISODES}] "
            f"Task={task_idx} | "
            f"AvgStepReward(inner)={avg_inner_return:.4f} | "
            f"WeightΔ={total_diff:.6f} | ε={epsilon:.3f}"
        )

    # Save final meta-trained initialization
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    torch.save(meta_agent.model.state_dict(), MODEL_PATH)
    print(f"\n[+] Meta-trained LSTM-Q (Reptile) saved to: {MODEL_PATH}")


if __name__ == "__main__":
    reptile_meta_training()
