# Assignment 4, Part 2: Deep Q-Network (DQN)

## Objective
This assignment focuses on implementing Deep Q-Network (DQN) for continuous state spaces using neural networks. 

You will work with two classic control environments:
- **Mountain Car**: A discrete-action task where you must control the car’s acceleration to build momentum and reach the goal flag at the top of a hill. For more details about the environment, please refer to the [Gymnasium](https://gymnasium.farama.org/environments/classic_control/mountain_car/) documentation.
- **Acrobot**: A more challenging environment where you must apply torques to a two-link pendulum system in order to swing it upward and reach a target height. For more details about the environment, please refer to the [Gymnasium](https://gymnasium.farama.org/environments/classic_control/acrobot/) documentation.

Your task is to implement the core DQN algorithm components and come up with good hyperparameters on which we will test your implementation on various environments.


## Files Provided

* `submission.py` – **Your main implementation file** (implement DQN components here)
* `requirements.txt` – Dependencies for the assignment

## Implementation Requirements

Complete the missing code in `submission.py`. You need to implement:

1. **QNetwork**: Neural network architecture for Q-function approximation
2. **ReplayBuffer**: Experience replay buffer for stable training  
3. **DQNAgent**: Main DQN algorithm with:
   - Epsilon-greedy exploration
   - Experience replay
   - Target network updates
   - Training your agent with Q-learning

## Environments
Your code should be agnostic to the environment. Your code will be tested on the following environments.
- **Mountain Car**: You are provided with an evaluation function `create_and_test_dqn_agent_on_mountain_car` that will test your code and visualize your agent's performance under `visualizations` folder. Uncomment it and run `submission.py` to test your implementation. During evaluation, your agent will be trained using 1000 episodes on this environment with batch size of 64 and the default parameters in your `DQNAgent` class.
- **Acrobot**: You are provided with an evaluation function `create_and_test_dqn_agent_on_acrobot` which will test your implementation on the Acrobot environment, save your model at `trained_acrobot_dqn.pth`, and visualize your agent's performance under `visualizations` folder. Provide this model in your submission file. 

## Grading
Your code will be evaluated based on the following criteria: 
- Correctly implementing QNetwork: 5 points.
- Correctly implementing ReplayBuffer: 5 points.
- Your agent's performance on Mountain Car trained with 1000 episodes: 10 points, a reward average of -110 or higher gives a full score. 
- Your agent's performance on three hidden environments, each trained with 1000 episodes: 15 points. 
- Your agent's performance on Acrobot using your pretrained model: 15 points. A reward average of -80 or higher gives a full score.

## Notes
- Do not make changes to the names of given files, functions, or classes. 
- Use **Python 3.9 or 3.10** with the provided `requirements.txt`
- Your code should complete training under 5 minutes on standard hardware.
- Your submission will take 4 time your runtime to be evaluated on gradescope (15-20 minutes).


## Submission
- Submit **`submission.py`**, **`requirements.txt`** and the **`trained_acrobot_dqn.pth`** model file to gradescore.
- You are provided with `Assignment 4: Part 2A` and `Assignment 4: Part 2B`. Submit the same code for both assignments. The first will evaluate your code only on the Mountain Car environment, while the latter will evaluate it on the Acrobot environment.
- You do not need to submit the visualization folder.

Good luck!
