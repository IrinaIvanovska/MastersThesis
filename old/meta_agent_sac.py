import numpy as np
import torch
from citylearn.citylearn import CityLearnEnv
from citylearn.wrappers import NormalizedObservationWrapper, StableBaselines3Wrapper
from collections import defaultdict


# Simple rule-based policy that returns actions for multiple buildings
def rbc_policy(observation, action_space, num_buildings=9):
    """
    Simple rule based policy based on day or night time
    """
    # Action for each building (initialize to zero)
    actions = np.zeros(num_buildings)
    
    hour = observation[2]  # Hour index is 2 for all observations
    
    for i in range(num_buildings):
        if 9 <= hour <= 21:
            # Daytime: release stored energy for each building
            actions[i] = -0.08
        elif (1 <= hour <= 8) or (22 <= hour <= 24):
            # Early nightime: store DHW and/or cooling energy for each building
            actions[i] = 0.091

        # Ensure the action is within the bounds of action_space (as action_space is continuous)
        actions[i] = np.clip(actions[i], action_space.low[i], action_space.high[i])  # Clip to action space bounds
    
    return actions


class MetaRLAgent:
    """
    Meta-Reinforcement Learning Agent using Rule-Based Policy and Meta-Learning Adaptation
    """

    def __init__(self, action_space, observation_space, num_buildings=9):
        self.action_space = action_space
        self.observation_space = observation_space
        self.num_buildings = num_buildings  # Set the number of buildings
        self.models = defaultdict(lambda: None)  # For task-specific models (e.g., one per building)
        self.meta_model = None  # Placeholder for meta-learner, like PEARL or MAML

    def register_reset(self, observation, action_space, agent_id):
        """Initialize the agent, adapt the model to the task and return action"""
        self.action_space = action_space  # Use the action space directly
        return self.compute_action(observation, agent_id)

    def compute_action(self, observation, agent_id):
        """Compute action using rule-based policy or learned adaptation"""
        # Rule-based policy
        actions = rbc_policy(observation, self.action_space, num_buildings=self.num_buildings)
        
        # If you had a meta-model, you could compute the action based on adaptation to the task
        if self.meta_model:
            pass  # Can implement meta-learning adaptation here, using the meta-model
        
        return actions

    def adapt_to_task(self, task_data):
        """Adapt the agent to a new task using the provided task data."""
        task_name = task_data["task_name"]
        observations = task_data["observations"]
        
        # Example: Adaptation could involve training or fine-tuning on task-specific data
        if self.models[task_name] is None:
            self.models[task_name] = "Model"  # Initialize a new model for the task
        
        # Fine-tune or adapt the model using task data (could be gradient-based)
        
        return self.models[task_name]

    def meta_train(self, env, num_iterations=1000, episodes=2):
        """Meta-training loop similar to model.learn"""
        for iteration in range(num_iterations):
            # Loop over episodes and adapt to new task at the start of each episode
            for episode in range(episodes):
                observation, _ = env.reset()  # Get initial observation
                done = False
                
                while not done:
                    # Compute the action using the agent
                    action = self.compute_action(observation, agent_id=0)
                    
                    # Take a step in the environment
                    observation, reward, done, info, _ = env.step(action)  # Unpack all 5 values from step
                   
                # After each episode, evaluate and adapt the model to the new task
                # (e.g., use meta-learning algorithms to adapt after training on this task)

    def evaluate(self, env):
        """Evaluate the agent's performance in the environment"""
        observation, _ = env.reset()  # Get initial observation
        done = False
        total_reward = 0
        
        while not done:
            action = self.compute_action(observation, agent_id=0)  # Use trained policy
            observation, reward, done, info, _ = env.step(action)  # Unpack all 5 values from step
            total_reward += reward
        
        return total_reward
