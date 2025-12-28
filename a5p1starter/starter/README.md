# Assignment 5, Part 1

## Objective

This assignment uses the  **[`MountainCarContinuous-v0`](https://gymnasium.farama.org/environments/classic_control/mountain_car_continuous/)** environment from **[Gymnasium](https://gymnasium.farama.org/)**. Your task is to implement two policy gradient agents **REINFORCE** and **REINFORCE with a learned Baselines**. Both the agents need to learn a stochastic policy to solve the task by directly interacting with the environment. They do not have direct access to the environment model. 

## Files Provided

- `submission.py` – Starter code (complete all TODOs here). **This is the only file you will submit.**
- `main.py` – Script to train and evaluate your agent locally.
- `requirements.txt` – Dependencies for the starter code.
- `README.md` – This file.

## Implementation Requirements

Complete the missing code in `submission.py`. 

To validate your agents, you can use 
```
python main.py              #vanilla reinforce
python main.py --baseline   #reinforce + baseline
```

## Notes

- **Do not** change the names of provided files, functions, or classes.
- Use **Python 3.9 or 3.10**. Your code should work with the package versions in `requirements.txt`.
- To receive credit, ensure your code **terminates within 20 minutes** under the Gradescope autograder (most correct solutions finish in a few minutes).
- Use **Gymnasium** (`import gymnasium as gym`), not legacy `gym`.
- Environment ID: `MountainCarContinuous-v0`.
- Seed `numpy`, `torch`, and the environment.
- CPU-only; do not rely on GPUs.

- The autograder directly imports and instantiates your TrainConfig class from `submission.py` in order to create agents for testing.
Because of this, your TrainConfig must keep the same fields and constructor signature as provided in the starter code.
- Do not rename or remove any fields in `TrainConfig`.
Do not change the expected argument names in `TrainConfig(...)`.
You may add convenience methods or internal helpers, but the public dataclass fields must stay intact.

## Submission

- You only need to submit the **`submission.py`** file to Gradescope.
- Do **not** zip the entire project or rename the file. The autograder looks specifically for `submission.py`.

## Autograder Workflow

The autograder will:

1. Import your `submission.py`.
2. Train both REINFORCE agents **from scratch** using your code.
3. Evaluate the agents on multiple seeds and slightly modified environment configurations (e.g., episode length).
4. Award credit based on average return, stability under small perturbations, and determinism.

## Optional Reading

You might have noticed that the Policy network provided in the starter code
makes several design choices regarding the form of action distribution,
observation preprocessing, etc. These design choices are important elements
of stabilizing the training of policy gradient algorithms. Many of them were
discovered empirically (e.g., by trying different architectures), and later
explained by theory. If you are interested in seeing more examples of such
design choices in a more complex policy gradient algorithm, you may enjoy
reading this blog post:

Huang, Shengyi, Rousslan Fernand Julien Dossa, Antonin Raffin, Anssi Kanervisto, and Weixun Wang. "The 37 implementation details of proximal policy optimization." The ICLR Blog Track 2023 (2022).
[link](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/)
