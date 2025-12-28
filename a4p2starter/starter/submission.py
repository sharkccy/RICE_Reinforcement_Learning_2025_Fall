import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import random
from collections import deque
from typing import Tuple, Optional
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import os   


class QNetwork(nn.Module):
    """
    Neural network for approximating Q-values.
    """
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dims: list = [128, 128]):
        """
        Initialize the Q-Network.
        
        Args:
            state_dim: Dimension of the state space
            action_dim: Number of discrete actions
            hidden_dims: List of hidden layer dimensions
        """
        super(QNetwork, self).__init__()
        
        ######## PUT YOUR CODE HERE ########
        # TODO: Define the neural network layers assuming the state is 1D and the action is discrete
        self.layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            # nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            # nn.BatchNorm1d(hidden_dims[1]),
            nn.GELU(),
            nn.Linear(hidden_dims[1], action_dim)
        )
        
        ######## PUT YOUR CODE HERE ########
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network.
        
        Args:
            x: Input tensor (batch_size, state_dim)
            
        Returns:
            Q-values for each action (batch_size, action_dim)
        """
        ######## PUT YOUR CODE HERE ########
        return self.layer(x)
        
        ######## PUT YOUR CODE HERE ########


class ReplayBuffer:
    """
    Experience replay buffer for DQN.
    """
    
    def __init__(self, capacity: int):
        """
        Initialize the replay buffer.
        
        Args:
            capacity: Maximum number of experiences to store
        """
        ######## PUT YOUR CODE HERE ########
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

        ######## PUT YOUR CODE HERE ########
    
    def push(self, state: np.ndarray, action: int, reward: float, 
             next_state: np.ndarray, done: bool):
        """
        Add an experience to the buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        ######## PUT YOUR CODE HERE ########
        # TODO: Add experience tuple to buffer
        
        self.buffer.append((state, action, reward, next_state, done))
        
        ######## PUT YOUR CODE HERE ########
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """
        Sample a batch of experiences from the buffer.
        
        Args:
            batch_size: Number of experiences to sample
            
        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        ######## PUT YOUR CODE HERE ########
        
        # TODO: Randomly sample batch_size experiences

        batch = random.sample(self.buffer, batch_size) # return lists of tuples (state, action, reward, next_state, done)
        states, actions, rewards, next_states, dones = zip(*batch) # * unzips the list of tuples to 5 lists as input to zip, zip to select the same index of each tuple, 
        return (torch.tensor(states, dtype=torch.float32),
                torch.tensor(actions, dtype=torch.long),
                torch.tensor(rewards, dtype=torch.float32),
                torch.tensor(next_states, dtype=torch.float32),
                torch.tensor(dones, dtype=torch.float32))
        
        ######## PUT YOUR CODE HERE ########
    
    def __len__(self) -> int:
        """Return the current size of the buffer."""
        ######## PUT YOUR CODE HERE ########
        # TODO: Return buffer size

        return len(self.buffer)
        
        ######## PUT YOUR CODE HERE ########


class DQNAgent:
    """
    Deep Q-Network agent implementation.
    Change your default hyperparameters to the ones you would like to use during evaluation
    """
    
    def __init__(self, state_dim: int, action_dim: int, batch_size: int = 64, device: str = 'cpu', 
                 buffer_size=1000, learning_starts=100):
        """
        Initialize the DQN agent.
        
        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            batch_size: Batch size for training
            device: Device to run on ('cpu' or 'cuda'). 
            buffer_size: Size of replay buffer. zero means no replay buffer is used
            learning_starts: Number of steps to start learning
        """
        
        ######## CHOSE YOUR HYPERPARAMETERS HERE ########
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = batch_size
        self.device = device
        self.buffer_size = buffer_size
        self.learning_starts = learning_starts
        self.gamma = 0.999
        #self.gamma = 0.999
        self.epsilon_start = 1.0
        self.epsilon_end = 0.05
        self.current_epsilon = self.epsilon_start
        self.epsilon_decay = 0.996       
        self.target_update_freq = 100
        self.lr = 1e-3
        #self.lr_decay = 0.999
        self.lr_decay = 0.999
        self.steps = 0
        
        
        ######## PUT YOUR CODE HERE ########
        # TODO: Initialize Q-networks (main and target), optimizer and replay buffer

        self.q_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network = QNetwork(state_dim, action_dim).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.optimizer = optim.AdamW(self.q_network.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=1, gamma=self.lr_decay)
        self.buffer = ReplayBuffer(self.buffer_size)
        
        ######## PUT YOUR CODE HERE ########
    
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select an action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training: Whether in training mode. if true, you should use epsilon-greedy policy, otherwise use greedy policy for inference.
            
        Returns:
            Selected action
        """
        ######## PUT YOUR CODE HERE ########

        if training:
            self.q_network.train()
            if np.random.rand() < self.current_epsilon:
                return np.random.randint(self.action_dim)
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                return int(torch.argmax(q_values).item())
        else:
            self.q_network.eval()
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            with torch.no_grad():
                q_values = self.q_network(state_tensor)
            return int(torch.argmax(q_values).item())
        
        ######## PUT YOUR CODE HERE ########
    
    def train_step(self) -> Optional[float]:
        """
        Perform one training step if enough experiences in buffer.
        
        Returns:
            Training loss (or None if no experiences in buffer)
        """
        if len(self.buffer) < self.batch_size:
            return None
        
        ######## PUT YOUR CODE HERE ########
        # TODO: Sample batch from replay buffer
        # TODO: Compute Q-values for current states
        # TODO: Compute target Q-values using target network
        # TODO: Compute loss and update main network
        # TODO: Update target network periodically
        # TODO: Decay epsilon
        

        states, actions, rewards, next_states, done = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        done = done.to(self.device)
        q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1) 
        # eg: self.q_nextwork = [[q1,q2,q3],[q1,q2,q3],...] , actions = [0,2,1,...] => gather to get [[q1],[q3],[q2],...], then squeeze to remove extra dim to get [q1,q3,q2,...]
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]  
            # max(1) take max along action_dim, and returns (values, indices), we want values only (0)
            target_q_values = rewards + self.gamma * next_q_values * (1 - done)
            # done is 1 if episode ended, so (1 - done) is 0 if episode ended, so no future reward added

        loss = F.mse_loss(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.target_update_freq > 0 and self.steps % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        return loss.item()
        ######## PUT YOUR CODE HERE ########
    
    def train_episode(self, env) -> Tuple[float, Optional[float]]:
        """
        Train the agent for one episode.
        This method is called by the training framework.
        Args:
            env: Gymnasium environment
        
        Returns:
            Tuple of (episode_return, loss)
        """        
        state, _ = env.reset()
        episode_return = 0.0
        total_loss = 0.0
        loss_count = 0
        
        # TODO: Implement the training loop and return the episode total reward and average loss
        
        
        done = False
        while not done:
            action = self.select_action(state, training=True)
            next_state, reward, teminated, truncated, _ = env.step(action)
            done = teminated or truncated
            self.buffer.push(state, action, reward, next_state, done)
            state = next_state
            episode_return += reward
            self.steps += 1
            
            if self.steps >= self.learning_starts:
                loss = self.train_step()
                if loss is not None:
                    total_loss += loss
                    loss_count += 1

        self.scheduler.step()
        self.current_epsilon = max(self.epsilon_end, self.current_epsilon * self.epsilon_decay)
        avg_loss = total_loss / loss_count if loss_count > 0 else None
        
        return episode_return, avg_loss
    
    def get_hyperparameters(self) -> dict:
        """Return hyperparameters for saving with model."""
        return {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'gamma': self.gamma,
            'epsilon_start': self.epsilon_start,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'current_epsilon': self.current_epsilon,
            'batch_size': self.batch_size,
            'target_update_freq': self.target_update_freq,
            'buffer_size': self.buffer_size,
            'learning_starts': self.learning_starts,
            'device': self.device,
            'lr': self.lr,
            'steps': self.steps
        }

    def load_weights(self, model_path: str):
        """Load weights from a model file."""
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        hyperparams = checkpoint['hyperparameters']
        
        # loading network weights: leaving this for your reference but feel free to change it
        
        self.state_dim = hyperparams['state_dim']
        self.action_dim = hyperparams['action_dim']
        self.gamma = hyperparams['gamma']
        self.epsilon_start = hyperparams['epsilon_start']
        self.epsilon_end = hyperparams['epsilon_end']
        self.epsilon_decay = hyperparams['epsilon_decay']
        self.batch_size = hyperparams['batch_size']
        self.target_update_freq = hyperparams['target_update_freq']
        self.buffer_size = hyperparams['buffer_size']
        self.learning_starts = hyperparams['learning_starts']
        self.lr = hyperparams['lr']
        
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        

def create_and_test_dqn_agent_on_mountain_car(batch_size: int = 64, 
                                              device: str = 'cpu', 
                                              visualize_agent: bool = False, 
                                              plot_losses: bool = True) -> DQNAgent:
    """
    You are provided with a function to create a DQN agent and test it on the MountainCar-v0 environment before submitting it to gradescore.
    The evaluation code will call your model directly and test on various environments.
    
    Args:
        batch_size: Batch size for training
        device: Device to run on ('cpu' or 'cuda').
        visualize_agent: If True, saves GIFs of agent learning progress
        
    Returns:
        Configured DQNAgent
    """
    
    env = gym.make('MountainCar-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, batch_size, device)
    
    # save loss for plots
    training_history = {
        'episodes': [],
        'returns': [],
        'losses': [],
        'eval_returns': []
    }
    recorded_episodes = {}
    # train the agent for 100 episodes
    print(f"Starting training... Learning will begin after {agent.learning_starts} steps")
    print(f"Initial epsilon: {agent.current_epsilon:.3f}")
    
    if visualize_agent:
        os.makedirs('visualizations', exist_ok=True)
    
    for episode in range(1000):
        episode_return, avg_loss = agent.train_episode(env)
        training_history['episodes'].append(episode+1)
        training_history['returns'].append(episode_return)
        training_history['losses'].append(avg_loss if avg_loss is not None else 0)
        
        if episode % 10 == 0:
            # test the agent
            eval_env = env if not visualize_agent else gym.make('MountainCar-v0', render_mode="rgb_array")
            
            state, _ = eval_env.reset()
            done = False
            total_reward = 0
            frames = []
            max_position = state[0]
            
            # Record this episode if it's every 100 episodes
            should_record = visualize_agent and episode % 100 == 0
            
            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                
                max_position = max(max_position, next_state[0])
                
                # Record frame for GIF
                if should_record:
                    frames.append(eval_env.render())
                
                state = next_state  
                total_reward += reward
            
            # Save GIF if this was a recorded episode
            if should_record and frames:
                gif_filename = f"visualizations/episode_{episode:03d}_reward_{total_reward:.3f}.gif"
                imageio.mimsave(gif_filename, frames, duration=0.05)
                print(f"Saved episode {episode+1} with reward {total_reward} to {gif_filename}")
            
            if visualize_agent and eval_env != env:
                eval_env.close()
                
            training_history['eval_returns'].append(total_reward)
            
            # Print progress
            learning_status = "Learning" if agent.steps >= agent.learning_starts else "Exploring"
            print()
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "0.0000"
            print(f"Episode {episode+1}: Return={episode_return:.1f}, Loss={loss_str}")
            print(f"Eval: Total reward={total_reward:.1f}, Steps={agent.steps}, Epsilon={agent.current_epsilon:.3f}, Status={learning_status}, Learning_rate={agent.scheduler.get_last_lr()[0]:.6f}")
        elif episode % 5 == 0:
            learning_status = "Learning" if agent.steps >= agent.learning_starts else "Exploring"
            loss_str = f"{avg_loss:.4f}" if avg_loss is not None else "0.0000"
            print(f"Episode {episode+1}: Return={episode_return:.1f}, Loss={loss_str}")
            print(f"Steps={agent.steps}, Epsilon={agent.current_epsilon:.3f}, Status={learning_status}, Learning_rate={agent.scheduler.get_last_lr()[0]:.6f}")
    
    env.close()
    
    # plot the training history 
    if plot_losses:
        plt.figure(figsize=(12, 4))
    
        plt.subplot(1, 3, 1)
        plt.plot(training_history['episodes'], training_history['returns'], label='Training Return')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Training Returns')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(training_history['episodes'], training_history['losses'], label='Training Loss')
        plt.xlabel('Episode')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        eval_episodes = [i*10 + 1 for i in range(len(training_history['eval_returns']))]
        plt.plot(eval_episodes, training_history['eval_returns'], label='Evaluation Return', marker='o')
        plt.xlabel('Episode')
        plt.ylabel('Return')
        plt.title('Evaluation Returns')
        plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    return agent


def create_and_test_dqn_agent_on_acrobot(batch_size: int = 64,
                                         device: str = 'cpu',
                                         visualize_agent: bool = False,
                                         plot_losses: bool = True,
                                         model_path: str = "trained_acrobot_dqn.pth",
                                         train: bool = True) -> DQNAgent:
    """
    You are provided with a function to create a DQN agent and test it on the Acrobot-v1 environment before submitting it to gradescore.
    The evaluation code will call your model directly and test on various environments.
    
    Args:
        batch_size: Batch size for training
        device: Device to run on ('cpu' or 'cuda').
        visualize_agent: If True, saves GIFs of agent learning progress
        plot_losses: If True, plots the training losses
        model_path: Path to the saved model
        train: If True, trains the agent, otherwise loads the model and evaluates it
    Returns:
        Configured DQNAgent
    """

    
    env = gym.make('Acrobot-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = DQNAgent(state_dim, action_dim, batch_size, device)
    
    if not train:
        assert model_path is not None, "model_path must be provided if train is False"
        assert os.path.exists(model_path), "model_path does not exist"
        agent.load_weights(model_path)

        # Print training history summary for confirmation
        # checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
        # training_history = checkpoint.get('training_history', None)
        # if training_history:
        #     print("\n=== Training History Summary ===")
        #     print(f"Episodes: {len(training_history['episodes'])}")
        #     print(f"Final training return: {training_history['returns'][-1] if training_history['returns'] else None}")
        #     print(f"Final eval return: {training_history['eval_returns'][-1] if training_history['eval_returns'] else None}")
        #     print(f"Last 5 training returns: {training_history['returns'][-5:] if len(training_history['returns']) >= 5 else training_history['returns']}")
        #     print(f"Last 5 eval returns: {training_history['eval_returns'][-5:] if len(training_history['eval_returns']) >= 5 else training_history['eval_returns']}")
    
        
        
        
    
    # save loss for plots
    training_history = {
        'episodes': [],
        'returns': [],
        'losses': [],
        'eval_returns': []
    }
    # train the agent for 100 episodes
    print(f"Starting training... Learning will begin after {agent.learning_starts} steps")
    print(f"Initial epsilon: {agent.current_epsilon:.3f}")
    
    if visualize_agent:
        os.makedirs('visualizations', exist_ok=True)
    
    for episode in range(1000):
        if train:
            episode_return, avg_loss = agent.train_episode(env)
            training_history['episodes'].append(episode+1)
            training_history['returns'].append(episode_return)
            training_history['losses'].append(avg_loss if avg_loss is not None else 0)
        
        if episode % 10 == 0:
            # test the agent
            eval_env = env if not visualize_agent else gym.make('Acrobot-v1', render_mode="rgb_array")
            
            state, _ = eval_env.reset()
            done = False
            total_reward = 0
            frames = []
            max_position = state[0]
            
            # Record this episode if it's every 100 episodes
            should_record = visualize_agent and episode % 100 == 0
            
            while not done:
                action = agent.select_action(state, training=False)
                next_state, reward, terminated, truncated, _ = eval_env.step(action)
                done = terminated or truncated
                
                max_position = max(max_position, next_state[0])
                
                # Record frame for GIF
                if should_record:
                    frames.append(eval_env.render())
                
                state = next_state  
                total_reward += reward
            
            # Save GIF if this was a recorded episode
            if should_record and frames:
                gif_filename = f"visualizations/acrobot_episode_{episode:03d}_reward_{total_reward:.1f}.gif"
                imageio.mimsave(gif_filename, frames, duration=0.05)
                print(f"Saved episode {episode+1} with reward {total_reward} to {gif_filename}")
            
            if visualize_agent and eval_env != env:
                eval_env.close()
                
            training_history['eval_returns'].append(total_reward)
            
            # Print progress
            learning_status = "Learning" if agent.steps >= agent.learning_starts else "Exploring"
            print(f"Episode {episode+1}: Return={total_reward:.1f}")
            print(f"Eval: Total reward={total_reward:.1f}, Steps={agent.steps}, Epsilon={agent.current_epsilon:.3f}, Status={learning_status}, Learning_rate={agent.scheduler.get_last_lr()[0]:.6f}")
        elif episode % 5 == 0:
            learning_status = "Learning" if agent.steps >= agent.learning_starts else "Exploring"
            print(f"Episode {episode+1}: Return={total_reward:.1f}")
            print(f"Steps={agent.steps}, Epsilon={agent.current_epsilon:.3f}, Status={learning_status}, Learning_rate={agent.scheduler.get_last_lr()[0]:.6f}")
    
    env.close()
    
    
    # plot the training history 
    if plot_losses:
        plt.figure(figsize=(12, 4))

        if 'returns' in training_history and len(training_history['returns']) > 0:
            plt.subplot(1, 3, 1)
            plt.plot(training_history['episodes'], training_history['returns'], label='Training Return')
            plt.xlabel('Episode')
            plt.ylabel('Return')
            plt.title('Training Returns')
            plt.legend()
        
        if 'losses' in training_history and len(training_history['losses']) > 0:
            plt.subplot(1, 3, 2)
            plt.plot(training_history['episodes'], training_history['losses'], label='Training Loss')
            plt.xlabel('Episode')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
        
        if 'eval_returns' in training_history and len(training_history['eval_returns']) > 0:
            plt.subplot(1, 3, 3)
            eval_episodes = [i*10 + 1 for i in range(len(training_history['eval_returns']))]
            plt.plot(eval_episodes, training_history['eval_returns'], label='Evaluation Return', marker='o')
            plt.xlabel('Episode')
            plt.ylabel('Return')
            plt.title('Evaluation Returns')
            plt.legend()
        
        plt.tight_layout()
        plt.show()
    
    print(f"Final performance: {training_history['eval_returns'][-1] if training_history['eval_returns'] else -500:.1f}")
    
    # Save the trained model
    if train and model_path is not None:
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        
        torch.save({
            'q_network_state_dict': agent.q_network.state_dict(),
            'target_network_state_dict': agent.target_network.state_dict(),
            'optimizer_state_dict': agent.optimizer.state_dict(),
            'hyperparameters': agent.get_hyperparameters(),
            'training_history': training_history,
            'final_performance': training_history['eval_returns'][-1] if training_history['eval_returns'] else -500
        }, model_path)
        
        print(f"Model saved to: {model_path}")
        
    
    return agent

if __name__ == "__main__":
    visualize_agent = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    print("Creating and testing DQN agent on Mountain Car. Uncomment the line below to test your implementation.")
    # agent = create_and_test_dqn_agent_on_mountain_car(visualize_agent=visualize_agent, device=device)
    
    
    print("Creating and testing DQN agent on Acrobot. Uncomment the line below to test your implementation.")
    agent = create_and_test_dqn_agent_on_acrobot(visualize_agent=visualize_agent, train=True, model_path="models/trained_acrobot_dqn.pth", device=device)
    
    # load and evaluate the model
    # agent = create_and_test_dqn_agent_on_acrobot(visualize_agent=visualize_agent, train=False, model_path="models/trained_acrobot_dqn.pth", device=device)
    # agent = create_and_test_dqn_agent_on_acrobot(visualize_agent=visualize_agent, train=False, model_path="trained_acrobot_dqn.pth", device=device)
    
    print("Agent created and tested successfully")