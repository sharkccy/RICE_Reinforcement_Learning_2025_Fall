import torch
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score
from submission_v2 import CustomDataset, TwoLayerNN, INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE

# 讀取全部 train data
dataset = CustomDataset('train.csv')
X = dataset.X
y = dataset.y

device = "cuda" if torch.cuda.is_available() else "cpu"

# 讀取 model
model = TwoLayerNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
model.load_state_dict(torch.load('model\\0.498\\best_model_79.pth', map_location=device))
# model.load_state_dict(torch.load('model\\0.553\\best_model_93.pth', map_location=device))
# model.load_state_dict(torch.load('model\\0.629\\best_model_50.pth', map_location=device))
model.to(device)
model.eval()

with torch.no_grad():
    X = X.to(device)
    y = y.to(device)
    outputs = model(X)
    _, predicted = torch.max(outputs, 1)

# 計算指標
acc = accuracy_score(y.cpu(), predicted.cpu())
precision = precision_score(y.cpu(), predicted.cpu(), average='macro', zero_division=0)
recall = recall_score(y.cpu(), predicted.cpu(), average='macro', zero_division=0)
f1 = f1_score(y.cpu(), predicted.cpu(), average='macro', zero_division=0)
cm = confusion_matrix(y.cpu(), predicted.cpu())

print(f"Train Acc: {acc}")
print(f"Precision: {precision}, Recall: {recall}, F1: {f1}")
# print("Confusion Matrix:")
# print(cm)