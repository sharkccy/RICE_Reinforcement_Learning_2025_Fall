# Assignment 3, Part 3

## Objective
In this assignment, your goal is to extend the active learning setup from 
Assignment 2, Part 3 by replacing the oracle with the model itself.
Instead of querying a fixed oracle for labels, your learner will use its own 
predictions as pseudo-labels — similar to how deep Q-learning uses its own 
predictions as training targets. 

## Files Provided
* `submission.py`: Starter code where you will implement the self-oracle and model update logic
* `train_supervised.csv`: Supervised training data (includes input features and target output)
* `train_unsupervised.csv`: Unsupervised data (includes only input features)
* `test.csv`: Test set to evaluate your solution. The autograder will use a different test set
* `requirements.txt`: Packages besides the Python Standard Library used in the started code

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
You will not be graded based on the performance of your final model. Instead, 
your submission will be evaluated by comparing its predictions on a test set 
against those produced by the reference solution. Because the training process 
is seeded for reproducibility, your predictions should exactly match the 
reference output if your implementation is correct.