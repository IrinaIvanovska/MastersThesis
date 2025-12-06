#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Unified reward function for consistent training and evaluation.
# ensures Meta-RL, Standard RL, and Evaluation all use the same objective.



def compute_reward_batch(df):
    """
    Compute reward for a batch of timesteps (DataFrame).
    Used during training for efficient batch processing.
    
    df with building state columns
    """
    return -(
        0.5 * df['average_unmet_cooling_setpoint_difference'].abs() +
        0.25 * df['cooling_demand'] +
        0.25 * df['heating_demand']
    )


def compute_reward_single(row):
    """
    reward for a single timestep (without action).
    Used for backward compatibility.
    row: df row or dict with building state
    """
    return -(
        0.5 * abs(float(row['average_unmet_cooling_setpoint_difference'])) +
        0.25 * float(row['cooling_demand']) +
        0.25 * float(row['heating_demand'])
    )


def compute_reward_with_action(row, action_continuous):
    """
    Compute reward considering how action affects energy consumption.
    training agents to select good actions.
    df row with building state
        action_continuous: Continuous action in [-1.0, 1.0]
            Negative: Reduce cooling, increase heating
            Positive: Increase cooling, reduce heating
    """
    # Simulate action effect on demands (matches evaluation KPI calculation)
    cooling_adj = float(row['cooling_demand']) * (1 + 0.1 * action_continuous)
    heating_adj = float(row['heating_demand']) * (1 - 0.1 * action_continuous)

    # Reward based on adjusted demands + setpoint tracking
    reward = -(
        0.5 * abs(float(row['average_unmet_cooling_setpoint_difference'])) +
        0.25 * cooling_adj +
        0.25 * heating_adj
    )

    # Small penalty for large actions (encourages conservative control)
    action_penalty = 0.01 * (action_continuous ** 2)

    return reward - action_penalty
