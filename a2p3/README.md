# Assignment 2, Part 3

## Objective
In this assignment, your goal is to train a machine learning model that predicts the target (model output) given a set of features (model input). You are given:
- a **supervised dataset**,
- an **unsupervised dataset**, and
- an **oracle** that provides the target output given an input.

Your access to the oracle is limited, and you can query it only 100 times. As such, you need to implement an exploration strategy that strategically chooses on which samples to query the oracle.

## Files Provided

* `submission.py`: Starter code where you will implement your exploration strategy
* `oracle.py`: The oracle. Do not change this file.
* `train_supervised.csv`: Supervised training data (includes input features and target output)
* `train_unsupervised.csv`: Unsupervised data (includes only input features)
* `test.csv`: Test set to evaluate your solution. The autograder will use a different test set.
* `requirements.txt`: Packages besides the Python Standard Library used in the started code.

## Problem
Your task is to:
1. **Train a model** on the supervised dataset (similar to Assignment 1, Part 2)
2. **Implement an exploration strategy** to select the most informative samples from the unsupervised dataset
3. **Query the oracle** for labels (up to 100 selected samples in total)
4. **Retrain your model** by incorporating the newly labeled data

Hints: 
* To learn the initial model, you can utilize a similar model architecture as Assignment 1, Part 2.
* You need to strategically choose which samples to label from the unsupervised set.

## Implementation Requirements

Complete the following components in `submission.py`:

1. **`ActiveLearner` class**: Implement your exploration strategy
   - `__init__`: Initialize your PyTorch model and strategy parameters
   - `train_initial_model`: Train neural network on initial supervised dataset
   - `predict`: Make predictions using trained neural network

2. **`active_learning_loop`**: Main active learning loop that orchestrates the process of exploration (i.e., selecting samples from the unsupervised data), querying the oracle, and retraining your model.

## Notes
* Do not make changes to the names of given files, functions, or classes.
* Your changes should be limited to the blocks of type
```
# --- SUBMISSION CODE: START ---


# --- SUBMISSION CODE: END ---
```
* Your code should work with the package versions provided in the requirements.txt file
* To receive credit, please ensure that your code terminates within 10 minutes when evaluated using the Gradescope autograder. Efficient solutions take much less time.


## Submission

Submit your completed `submission.py` file to Gradescope. The autograder will:
1. Load your code and train the model on the fly
2. Run your active learning loop with access to the oracle
3. Evaluate your final model on the hidden test set


## Grading

Your submission will be evaluated based on correctness, model accuracy (of both initial and final models), and the exploration strategy. A two layer network is expected to have an initial accuracy of around 50% and improve to 70-80% after active learning.