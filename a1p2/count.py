import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

data = np.loadtxt('train.csv', delimiter=',', skiprows=1)
num_of_each_class = np.bincount(data[:, -1].astype(int))
print(num_of_each_class)
