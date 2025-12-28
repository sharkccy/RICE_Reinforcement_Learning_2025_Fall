# Assignment 5, Part 2

## Objective

This assignment uses the  **[`BipedalWalker-v3`](https://gymnasium.farama.org/environments/box2d/bipedal_walker/)** environment from **[Gymnasium](https://gymnasium.farama.org/)**.
You have to train a policy using the DDPG algorithm implemented in the [Stable Baselines 3 (SB3)](https://stable-baselines3.readthedocs.io/en/master/) package.
As you know from the course readings, SB3 provides open-source implementations of deep RL algorithms in Python.
The purpose of this problem is to help you get familiar with the SB3 framework, by developing an agent for `BipedalWalker-v3`.
You will also gain experience of tuning hyperparameters for deep RL algorithms.

## Files Provided

- `submission.py` – Starter code.
- `requirements.txt` – Dependencies.
- `README.md` – This file.

## Implementation Requirements

Use the `submission.py` file as a starting point for your implementation.
Save your trained DDPG model, which is a `.zip` file (as explained in the next section).

## Your submission

- A `submission_model.zip` file. `.zip` format is used by the SB3 framework to save model weights (you can read the SB3 docs to know more about this), as well are some other training details. You file must be named `submission_model.zip`. Gradescope will automatically unzip your submitted zip file but that's ok.
- `submission.py` file, upload this file too with the above `submission_model.zip` file.

## Notes

- Use **Python 3.9 or 3.10**. Your code should work with the package versions in `requirements.txt`.
- Use **Gymnasium** (`import gymnasium as gym`), not legacy `gym`.
- Environment ID: `BipedalWalker-v3`.
- Seed `numpy`, `torch`, the environment, and any other randomness that you might be introducing. This ensures result reproducibility.

## Autograder Workflow
- Load your submitted `submission_model.zip` file into a SB3 DDPG class.
- Test A (implemented in Assignment 5, Part 2A): Evaluates your submitted model on multiple episodes of `BipedalWalker-v3`. Compute the mean reward across these episodes. And, provides a score based on this mean reward.
- Test B (implemented in Assignment 5, Part 2B): Evaluates the submitted model on multiple episodes of the hard variant of `BipedalWalker-v3`. Compute the mean reward across these episodes. And, provides a score based on this mean reward.
- You can upload a different `submission_model.zip` and `submission.py` files for Test A and Test B through corresponding Gradescope assignments.
