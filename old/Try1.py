import numpy as np
from citylearn.citylearn import CityLearnEnv
from collections import defaultdict
from copy import deepcopy

# --- Parameters ---
action_space = [-1, 0, 1]  # Discretized actions
state_bins = 10            # For discretizing continuous observations
alpha = 0.1                # Q-learning LR
gamma = 0.99               # Discount factor
meta_lr = 0.05             # Reptile meta update rate
episodes_per_task = 3
meta_iterations = 10

# --- State discretizer ---
def discretize_state(obs, bins=state_bins):
    return tuple(np.digitize(o, np.linspace(0, 1, bins)) for o in obs)

# --- Q-table initializer ---
def init_q_table():
    return defaultdict(lambda: np.zeros(len(action_space)))

# --- Task-specific Q-learning update ---
def q_learning(env, q_table, episodes):
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            state = discretize_state(obs)
            if np.random.rand() < 0.1:
                a_idx = np.random.choice(len(action_space))
            else:
                a_idx = np.argmax(q_table[state])
            action = [action_space[a_idx]] * len(env.action_space)
            next_obs, reward, done, _, _ = env.step(action)
            next_state = discretize_state(next_obs)
            q_target = reward + gamma * np.max(q_table[next_state])
            q_table[state][a_idx] += alpha * (q_target - q_table[state][a_idx])
            obs = next_obs
    return q_table

# --- Reptile meta-update ---
def reptile_update(meta_q, task_q, meta_lr):
    for state in task_q:
        if state not in meta_q:
            meta_q[state] = np.copy(task_q[state])
        else:
            meta_q[state] += meta_lr * (task_q[state] - meta_q[state])
    return meta_q

# --- Environment setup ---
def initialize_env(building_id):
    return CityLearnEnv(schema='citylearn_challenge_2023_phase_3_1', building_ids=[building_id], central_agent=False)

# --- Meta-training loop ---
def meta_train_q_learning(building_ids, episodes=3, meta_iters=10):
    meta_q = init_q_table()
    for meta_iter in range(meta_iters):
        for b_id in building_ids:
            env = initialize_env(b_id)
            task_q = deepcopy(meta_q)
            task_q = q_learning(env, task_q, episodes)
            meta_q = reptile_update(meta_q, task_q, meta_lr)
        print(f"Meta-iteration {meta_iter + 1}/{meta_iters} complete.")
    return meta_q

# --- Fine-tuning on unseen building ---
def fine_tune_and_evaluate(meta_q, building_id=5, episodes=3):
    env = initialize_env(building_id)
    q_table = deepcopy(meta_q)
    q_table = q_learning(env, q_table, episodes)

    obs, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        state = discretize_state(obs)
        a_idx = np.argmax(q_table[state])
        action = [action_space[a_idx]] * len(env.action_space)
        obs, reward, done, _, _ = env.step(action)
        total_reward += reward

    # KPIs
    kpis = env.evaluate()
    kpis = kpis.pivot(index="cost_function", columns="name", values="value").round(3).dropna(how="all")
    print(f"\nKPIs for Building {building_id}:")
    print(kpis)
    print(f"Total Reward: {total_reward}")
    return kpis

# === Run Training ===
train_buildings = [0, 1, 2, 3, 4]
meta_q_table = meta_train_q_learning(train_buildings, episodes=episodes_per_task, meta_iters=meta_iterations)

# === Evaluate ===
kpis = fine_tune_and_evaluate(meta_q_table, building_id=5, episodes=episodes_per_task)
