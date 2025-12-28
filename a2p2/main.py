import numpy as np
import matplotlib.pyplot as plt

# Import environments
from environments import ValidationEnv

# Import agents
from submission import MyAgent
from agents import RandomAgent

def run_experiment(env, agent, episodes=500):
    """
    Run one agent on one environment for a fixed number of episodes.
    Returns: rewards, cumulative rewards
    """
    rewards = []
    cumulative_rewards = []

    obs, _ = env.reset()
    total_reward = 0

    for ep in range(episodes):
        action = agent.select_action(obs)
        next_obs, reward, done, _, info = env.step(action)
        agent.update(obs, action, reward, next_obs)

        total_reward += reward
        rewards.append(reward)
        cumulative_rewards.append(total_reward)

        obs = next_obs
        if done:
            obs, _ = env.reset()

    return rewards, cumulative_rewards


def plot_results(results, title):
    """
    Plot learning curves for each agent.
    """
    fig, ax = plt.subplots(1, 1, figsize=(14, 5))

    # Plot cumulative rewards
    for name, data in results.items():
        ax.plot(data["cumulative_rewards"], label=name)
    ax.set_title(f"{title} - Cumulative Rewards")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Cumulative Rewards")
    ax.legend()
    ax.grid(True)

    plt.show()


if __name__ == "__main__":

    env = ValidationEnv(n_products=5, n_users=2)
    agents = {
        "RandomAgent": RandomAgent(env.n_products, env.n_users),
        "MyAgent": MyAgent(env.n_products, env.n_users)
    }
    results = {}
    for name, agent in agents.items():
        rewards, cumulative_rewards = run_experiment(env, agent, episodes=500)
        results[name] = {
            "rewards": rewards, "cumulative_rewards": cumulative_rewards}
    plot_results(results, "Validation Environment")
