# Assignment 2, Part 2

## Objective
This assignment simulates simplified **product recommendation** settings.
Your task is to implement an **agent** that maximizes its cumulative *reward*,
by effectively recommending products in these simplified settings.

## Problem Description

This assignment simulates **three** simplified product recommendation settings.

In each setting, your agent has to recommend products to users over multiple
steps of interaction. At each step of the interaction,
- the agent must select one product,
- this product is then recommended to the user, and
- the user then decides whether to "click" the product or not.

The step function of the enviroment simulates this interaction.
The step function
- takes agent's action as input,
- provides a reward based on whether the user "clicks" the product, and
- provides an observation about the user ID of the next user.

The agent's goal is to maximize its cumulative reward.

You will design **one** agent, which will be tested in the following three
simplified product recommendation settings:
- Fixed Click Probabilities
    - The click probabilities depend only on the recommended product.
    - As such, the click probabilities are identical for all users.
    - These conditional probabilities of clicks do not change with time.
- Time-Varying Click Probabilities
    - The click probabilities depend only on the recommended product.
    - These conditional probabilities of clicks change with time.
- User-Dependent Click Probabilities
    - The click probabilities depend on the product as well as the user.
    - These conditional probabilities of clicks do not change with time.

Each setting is implemented as a (hidden) environment,
using the `BaseEnv` class provided in your started code.
While you do not have access to the three hidden environments,
you can validate your agent in the provided `ValidationEnv`.


## Files Provided 

* `environments.py` – Base and validation environments that simulate simplified 
    product recommendation settings.
* `agents.py` – Provies a base class for agents; and, an example random agent.
* `submission.py` – Started code for implementing your agent.
* `main.py` – Script to run experiments, compare MyAgent vs RandomAgent, and generate plots.
* `requirements.txt` – Dependencies for the assignment's started code.
* `README.md` - This file.

## Implementation Requirements  

Complete the following in `submission.py`:
- **`MyAgent` class**  
    - `__init__`: Initialize algorithm parameters.
    - `select_action`: Choose an action given the observation.
    - `update`: Update the agent based on feedback.

You are free to implement **any algorithm**.

**Hint**: Compare this problem setting to the multi-armed bandits setting,
    which we studied in the class. It can serve as a useful starting point.

## Notes
- Do not make changes to the names of given files, functions, or classes.
- Use **Python 3.9 or 3.10**. Your code should work with the package versions
    provided in the `requirements.txt` file.
- Install dependencies:
  ```bash
  python3 -m venv bandit-env
  source bandit-env/bin/activate   # Linux/Mac
  bandit-env\Scripts\activate      # Windows

  pip install -r requirements.txt
  ```
- Do not rename `submission.py` or the `MyAgent` class.
    You may add helper functions or attributes inside `MyAgent`,
    but keep the class interface intact.
- To receive credit, please ensure that your code terminates within 20 minutes
    when evaluated using the Gradescope autograder.
    Efficient solutions take much less time.

## Submission

- You only need to submit the **`submission.py`** file to Gradescope.
- Do **not** zip the entire project or rename the file.
    The autograder will look specifically for `submission.py`.

## Autograder Workflow
The autograder will:
1. Import your implementation using:
   ```
   from submission import MyAgent
   ```
2. Run your agent on three (hidden) test environments.
3. Award credits based on your agent's performance.


