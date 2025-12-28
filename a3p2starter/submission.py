import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt

from helper import BaseAgent


class MonteCarloAgent(BaseAgent):
    def __init__(self, num_actions, env_step, env_reset):
        """
        Initializes the agent.
        """
        super().__init__(num_actions, env_step, env_reset)
        ######## PUT YOUR CODE HERE ########
        self.Q = {}
        self.epsilon = 0.1
        self.gamma = 0.99
        self.returns_sum = {}
        self.returns_count = {}
        # alpha = 1/n where n is the number of visits to (s,a)

        ######## PUT YOUR CODE HERE ########

    def make_policy(self):
        """
        Return a policy function that will be used for evaluation. The policy
        takes observation as input and returns an action.
        """
        def policy_func(obs):
            """
            Takes observation as input and returns an action.
            """
            ######## PUT YOUR CODE HERE ########

            q_values = [self.Q.get((obs, a), 0) for a in range(self.num_actions)]
            return int(np.argmax(q_values))
            
            ######## PUT YOUR CODE HERE ########
        return policy_func

    def make_behavior_policy(self):
        """
        Return a policy function that will be used for evaluation. The policy
        takes observation as input and returns an action.
        """
        def policy_func(obs):
            """
            Takes observation as input and returns an action.
            """
            ######## PUT YOUR CODE HERE ########

            if np.random.rand() < self.epsilon:
                return np.random.randint(self.num_actions)
            else:
                q_values = [self.Q.get((obs, a), 0) for a in range(self.num_actions)]
                return int(np.argmax(q_values))
            
            ######## PUT YOUR CODE HERE ########
        return policy_func

    def update(self, episode):
        """
        Update the agent given the data collected during the episode.
        """
        ######## PUT YOUR CODE HERE ########
        G = 0
        visited_state_actions = set()
        for t in reversed(range(len(episode))):
            obs, action, reward, _, _ = episode[t]
            G = self.gamma * G + reward
            if (obs, action) in visited_state_actions:
                continue
            visited_state_actions.add((obs, action))

            # Update returns sum and count
            self.returns_sum[(obs, action)] = self.returns_sum.get((obs, action), 0) + G
            self.returns_count[(obs, action)] = self.returns_count.get((obs, action), 0) + 1
            # Update Q value
            self.Q[(obs, action)] = self.returns_sum[(obs, action)] / self.returns_count[(obs, action)]
        
        ######## PUT YOUR CODE HERE ########

    def train(self, num_train_episodes, make_plot=False):
        list_returns = []
        list_eval_returns_mean = []
        list_eval_returns_stdev = []
        list_eval_episodes = []
        for episode_idx in range(num_train_episodes):
            # Generate an episode with behavior policy
            episode, episode_return = self.run_episode(self.behavior_policy)

            # Update the policy with the data collected during the episode
            self.update(episode)

            # Store the return
            list_returns.append(episode_return)

            # Evaluate the policy after every 1000 training episodes
            if episode_idx % 1000 == 0:
                mean_return, stdev_return = self.evaluate()
                list_eval_episodes.append(episode_idx)
                list_eval_returns_mean.append(mean_return)
                list_eval_returns_stdev.append(stdev_return)
                print(f"Episode #: {episode_idx}, Return Mean: {round(mean_return, 3)}, St.Dev.: {round(stdev_return, 3)}")

        if make_plot:
            plt.plot(list_eval_episodes, list_eval_returns_mean,
                marker='o', linestyle='-', color='blue', label='Mean')
            plt.fill_between(list_eval_episodes,
                np.array(list_eval_returns_mean) - np.array(list_eval_returns_stdev),
                np.array(list_eval_returns_mean) + np.array(list_eval_returns_stdev),
                color='blue', alpha=0.2, label='St. Dev.')
            plt.ylabel('Return')
            plt.xlabel('Episode#')
            plt.title('Performance during training')
            plt.legend()
            plt.show()


class QLearningAgent(BaseAgent):
    def __init__(self, num_actions, env_step, env_reset):
        """
        Initializes the agent.
        """
        super().__init__(num_actions, env_step, env_reset)
        ######## PUT YOUR CODE HERE ########
        self.Q = {}
        self.epsilon = 0.1
        self.alpha = 0.05
        self.gamma = 0.99
        ######## PUT YOUR CODE HERE ########

    def make_policy(self):
        """
        Return a policy function that will be used for evaluation. The policy
        takes observation as input and returns an action.
        """
        def policy_func(obs):
            """
            Takes observation as input and returns an action.
            """
            ######## PUT YOUR CODE HERE ########

            q_values = [self.Q.get((obs, a), 0) for a in range(self.num_actions)]
            return int(np.argmax(q_values))
        
            ######## PUT YOUR CODE HERE ########
        return policy_func

    def make_behavior_policy(self):
        """
        Return a policy function that will be used for evaluation. The policy
        takes observation as input and returns an action.
        """
        def policy_func(obs):
            """
            Takes observation as input and returns an action.
            """
            ######## PUT YOUR CODE HERE ########

            if np.random.rand() < self.epsilon:
                return np.random.randint(self.num_actions)
            else:
                q_values = [self.Q.get((obs, a), 0) for a in range(self.num_actions)]
                return int(np.argmax(q_values))
            
            ######## PUT YOUR CODE HERE ########
        return policy_func

    def update(self, obs, action, next_obs, reward):
        ######## PUT YOUR CODE HERE ########
        old_q = self.Q.get((obs, action), 0)

        next_q_values = [self.Q.get((next_obs, a), 0) for a in range(self.num_actions)]
        td_target = reward + self.gamma * max(next_q_values)

        td_delta = td_target - old_q
        self.Q[(obs, action)] = old_q + self.alpha * td_delta
        
        ######## PUT YOUR CODE HERE ########

    def train(self, num_train_episodes, make_plot=False):
        list_returns = []
        list_eval_returns_mean = []
        list_eval_returns_stdev = []
        list_eval_episodes = []
        for episode_idx in range(num_train_episodes):
            # Reset environment before beginning the episode
            done = False
            obs, _ = self.env_reset()
            episode_return = 0

            # Run the episode and update the policy 
            while not done:
                # Generate a step with behavior policy
                action = self.behavior_policy(obs)
                next_obs, reward, terminated, truncated, _ = self.env_step(
                    action)
                done = terminated or truncated

                # Second, update the policy
                self.update(obs, action, next_obs, reward)

                # Prepare for next step
                obs = next_obs
                episode_return += reward

            # Store the return
            list_returns.append(episode_return)

            # Evaluate the policy after every 1000 training episodes
            if episode_idx % 1000 == 0:
                mean_return, stdev_return = self.evaluate()
                list_eval_episodes.append(episode_idx)
                list_eval_returns_mean.append(mean_return)
                list_eval_returns_stdev.append(stdev_return)
                print(f"Episode #: {episode_idx}, Return Mean: {round(mean_return, 3)}, St.Dev.: {round(stdev_return, 3)}")

        if make_plot:
            plt.plot(list_eval_episodes, list_eval_returns_mean,
                marker='o', linestyle='-', color='blue', label='Mean')
            plt.fill_between(list_eval_episodes,
                np.array(list_eval_returns_mean) - np.array(list_eval_returns_stdev),
                np.array(list_eval_returns_mean) + np.array(list_eval_returns_stdev),
                color='blue', alpha=0.2, label='St. Dev.')
            plt.ylabel('Return')
            plt.xlabel('Episode#')
            plt.title('Performance during training')
            plt.legend()
            plt.show()
