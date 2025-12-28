"""
Starter code for Assignment 5, Part 2:

DDPG Trainer for BipedalWalker-v3 using stable-baselines3.
"""

import gymnasium as gym
import numpy as np
from typing import List, Tuple

from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.buffers import ReplayBuffer


class EpisodeRewardCallback(BaseCallback):
    """
    A custom callback to collect episodic total rewards (returns).
    
    This callback must be used with an environment wrapped
    in a `stable_baselines3.common.monitor.Monitor` wrapper.
    """
    def __init__(self, verbose: int = 0):
        """Initializes the callback."""
        super().__init__(verbose)
        self.episode_rewards: List[float] = []

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to env.step().
        
        It checks if an episode has just finished (by checking the `info`
        dict for the 'episode' key, which is added by the Monitor wrapper)
        and appends the episode's total reward to the `episode_rewards` list.
        """
        # We assume a single environment is being used
        # Check if the episode has finished
        if "episode" in self.locals["infos"][0]:
            ep_reward = self.locals["infos"][0]["episode"]["r"]
            self.episode_rewards.append(ep_reward)
            
            if self.verbose > 0:
                print(f"Episode {len(self.episode_rewards)} finished. Reward: {ep_reward:.2f}")
                
        return True


class DDPGTrainer:
    """
    A wrapper class to train a DDPG agent on BipedalWalker-v3.
    
    Provides a simple `train` method that returns a list of episode
    rewards.
    """
    
    def __init__(self, env: gym.Env):
        """
        Initializes the environment and the DDPG model.

        Args:
            env: A gymnasium environment.
        """
        # NOTE: 
        # SB3 DDPG implemention has a variable called `policy_kwargs`.
        # This variable is used to specify the policy architecture.
        # The starter code provides a specific value of this variable.
        # In your submission, this value MUST NOT be changed.
        # In the DDPG SB3 implementation for the policy_kwargs,
        # you should pass the value provided below:
        self.policy_kwargs=dict(net_arch=[512, 256])
        ######## PUT YOUR CODE HERE ########
        # (Optional) Initialize any necessary variables besides policy_kwargs
        pass
        ######## PUT YOUR CODE HERE ########

    def train(self, total_timesteps: int) -> List[float]:
        """
        Function which calls the SB3 DDPG trainer.

        This file contains a trainer class that can be imported and used to train a DDPG agent. The `train` method returns a list of episode returns collected during training. You will have to use the SB3 package to implement your trainer.

        Args:
            total_timesteps: The total number of timesteps that we want to train the DDPG model for.

        Returns:
            Returns a list of episode returns collected during training.
        """
        ######## PUT YOUR CODE HERE ########
        pass
        ######## PUT YOUR CODE HERE ########
        return reward_callback.episode_rewards

    def evaluate(self, n_eval_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluates the trained agent for a number of episodes.
        
        Args:
            n_eval_episodes: The number of episodes to evaluate.

        Returns:
            A tuple of (mean_reward, std_reward).
        """
        print(f"Evaluating agent over {n_eval_episodes} episodes...")
        mean_reward, std_reward = evaluate_policy(
            self.model, 
            self.env, 
            n_eval_episodes=n_eval_episodes
        )
        print(f"Evaluation complete: Mean reward = {mean_reward:.2f} +/- {std_reward:.2f}")
        return mean_reward, std_reward

    def save(self, path: str):
        """
        Saves the trained model to a file.
        
        Args:
            path: The path to save the model (e.g., 'ddpg_bipedalwalker.zip').

        Returns:
            None
        """
        self.model.save(path)
        print(f"Model saved to {path}")

    def load(self, path: str):
        """
        Loads a pre-trained model from a file.
        
        Args:
            path: The path to load the model from.

        Returns:
            None
        """
        self.model = DDPG.load(path, env=self.env)
        print(f"Model loaded from {path}")


if __name__ == "__main__":
    # Create the environment
    env_id = "BipedalWalker-v3"
    print(f"Creating environment: {env_id}")
    env = gym.make(env_id, hardcore = True, render_mode=None)

    # 1. Initialize the agent
    trainer = DDPGTrainer(env)
    
    # 2. Train the agent
    ######## PUT YOUR CODE HERE ########
    # (Optional) Change the number of training time steps
    training_timesteps = 100000
    ######## PUT YOUR CODE HERE ########
    episode_rewards = trainer.train(total_timesteps=training_timesteps)

    
    # 3. Show the results
    print(f"\n--- Training Results ---")
    if episode_rewards:
        print(f"Collected {len(episode_rewards)} episode rewards.")
        print(f"First 5 rewards: {[f'{r:.2f}' for r in episode_rewards[:5]]}")
        print(f"Last 5 rewards: {[f'{r:.2f}' for r in episode_rewards[-5:]]}")
        print(f"Average reward during training: {np.mean(episode_rewards):.2f}")
    else:
        print("No full episodes were completed in the short training run.")

    # 4. Evaluate the trained agent
    print("\n--- Evaluation Results ---")
    mean_rew, std_rew = trainer.evaluate(n_eval_episodes=5)
    
    # 5. Save the model
    model_path = "submission_model.zip"
    trainer.save(model_path)
    
    # 6. Load the model and evaluate again
    print("\n--- Loading and Re-evaluating ---")
    new_trainer = DDPGTrainer(env)
    new_trainer.load(model_path)
    new_trainer.evaluate(n_eval_episodes=5)