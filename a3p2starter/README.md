# Assignment 3, Part 2

## Objective
This assignment uses the [Frozen Lake](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) environment from [Gymnasium](https://gymnasium.farama.org/). Your task is to compute the optimal policy for this environment using value-based reinforcement learning algorithms. In Assignment 3 Part 1, you had access to the environment's model (transition and reward functions, or equivalently, the environment map). In contrast, in this Part 2, your agent will need to compute the policy without direct access to these variables by interacting with the environment. Specifically, you will implement two RL agents:
- MonteCarloAgent: Based on the On-policy first-visit MC control algorithm.
- QLearningAgent: Based on the Q-learning algorithm.

## Files Provided 

* `submission.py` – Starter code.
* `helper.py` – Helper functions.
* `main.py` – Script to train your agent.
* `requirements.txt` – Dependencies for the assignment's starter code.
* `README.md` - This file.

## Implementation Requirements  

Complete the missing code in `submission.py`.

## Notes
- Do not make changes to the names of given files, functions, or classes.
- Use **Python 3.9 or 3.10**. Your code should work with the package versions
    provided in the `requirements.txt` file.
- To receive credit, please ensure that your code terminates within 20 minutes
    when evaluated using the Gradescope autograder.
    Efficient solutions take much less time.

## Submission

- You only need to submit the **`submission.py`** file to Gradescope.
- Do **not** zip the entire project or rename the file.
    The autograder will look specifically for `submission.py`.

## Autograder Workflow
The autograder will:
- Import your implementation.
- Evaluate the implemented algorithms on multiple variants of the Frozen Lake environment, each with different map layouts.
- Award credits based on the accuracy of the computed policy and values.
