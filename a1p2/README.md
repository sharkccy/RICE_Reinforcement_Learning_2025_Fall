# Assignment 1, Part 2

## Objective
The goal of this assignment is to implement and train a 2-layer neural network using PyTorch. You will be working with a synthetic dataset for a classification task.

## Files Provided

* `submission.py`: Starter code, where you will implement your code. 
* `train.csv`: The training data.
* `README.md`: This file.

## Problem 
Implement and train a 2-layer neural network using PyTorch by completing the missing code in `submission.py` file. 

Specifically, you need to first:
* Complete the `TwoLayerNN` class.
* Complete the `train_model` function.

After completing the code, you need to:
* Train and save a neural network model by running `python submission.py`

## Notes
* Do not make changes to the names of given files, functions, or classes.
* Your changes should be limited to the blocks of type
```
# --- SUBMISSION CODE: START ---
# TODO: INSERT YOUR CODE HERE.


# --- SUBMISSION CODE: END ---
```
* Importing any additional packages (besides those in the starter code) is not allowed.
* The code has been tested on Python 3.13.7, PyTorch 2.7.1, and Numpy 2.3.2. In case your code does not compile or run, the recommended first step for troubleshooting is to try it on a virtual environment with the specified version of Python and associated packages.
* To receive credit, please ensure that your code terminates within 20 minutes when evaluated using the Gradescope autograder. Efficient solutions take much less time.

## Submission
Submit your completed `submission.py` and trained `model.pth` files to Gradescope.

## Grading
Your submission will be graded based on the accuracy of your trained `model.pth` on a hidden test set. Your code must run without errors and generate the `model.pth` file.