import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
# from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
# import matplotlib.pyplot as plt
# --- Constants ---
# NOTE: Do not change these.
BATCH_SIZE = 64
INPUT_SIZE = 4
HIDDEN_SIZE = 64
OUTPUT_SIZE = 3

# --- Helper Functions ---
class CustomDataset(Dataset):
    """A custom dataset class for loading the training data."""
    def __init__(self, file_path):
        # We use numpy to load the data
        data = np.loadtxt(file_path, delimiter=',', skiprows=1)
        
        # The last column is the target, the rest are features
        X = data[:, :-1]
        y = data[:, -1]
        
        # Convert to PyTorch tensors
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
        
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# --- Neural Network Definition ---
class TwoLayerNN(nn.Module):
    """A simple two-layer neural network."""
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the TwoLayerNN model. This should be a two-layer network,
        meaning it has one hidden layer.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output classes.
        """
        super(TwoLayerNN, self).__init__()
        # --- STUDENT CODE: START ---
        # TODO: INSERT YOUR CODE HERE.
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.LeakyReLU(),
            nn.Linear(hidden_size, output_size),
        )
        # --- STUDENT CODE: END ---
    
    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the network.
        """        
        # --- STUDENT CODE: START ---
        # TODO: INSERT YOUR CODE HERE.
        return self.layer(x)
        # --- STUDENT CODE: END ---

# --- Training Function ---
def train_model(model, train_loader):
    """
    This function trains the model on the provided data.
    
    Args:
        model (nn.Module): The neural network model to train.
        train_loader (DataLoader): The data loader for the training set.
    """
    # --- STUDENT CODE: START ---
    # TODO: INSERT YOUR CODE HERE.

    full_dataset = train_loader.dataset
    total_len = len(full_dataset)
    train_len = int(0.8 * total_len)
    val_len = total_len - train_len
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_len, val_len])
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    labels = []
    for i in range(len(train_dataset)):
        _, y = train_dataset[i]
        labels.append(y.item())
    labels = torch.tensor(labels)
    for cls in torch.unique(labels):
        count = (labels == cls).sum().item()
        # print(f"Class {cls.item()}: {count} samples in training set")

    num_epochs = 100
    init_lr = 1e-3
    lr_decay_factor = 0.95
    weight_decay = 1e-5
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    # print(f"Using device: {device}")


    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=init_lr, weight_decay=weight_decay)  # update weight acccording to the gradient got from loss.backward()
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay_factor) 
    scaler = torch.amp.GradScaler()

    training_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state = None

    for epoch in range(num_epochs):
        model.train()
        torch.cuda.empty_cache()
        # print(f"Epoch {epoch+1}/{num_epochs} started")
        total_train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type=device):
                outputs = model(batch_x)
                batch_loss = loss(outputs, batch_y)


            scaler.scale(batch_loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_train_loss += batch_loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        scheduler.step()
        training_losses.append(avg_train_loss)
        
        # print(f"Avg batch loss for epoch {epoch+1}: {avg_train_loss}")

        model.eval()
        with torch.no_grad():
            total_val_loss = 0.0
            all_val_y = []
            all_val_pred = []
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                val_outputs = model(batch_x)
                val_loss = loss(val_outputs, batch_y)
                total_val_loss += val_loss.item()
                _, predicted = torch.max(val_outputs, 1)
                all_val_y.extend(batch_y.cpu().numpy())
                all_val_pred.extend(predicted.cpu().numpy())

            avg_val_loss = total_val_loss / len(val_loader)
            val_losses.append(avg_val_loss)

        # acc = accuracy_score(all_val_y, all_val_pred)
        # precision = precision_score(all_val_y, all_val_pred, average='macro', zero_division=0)
        # recall = recall_score(all_val_y, all_val_pred, average='macro', zero_division=0)
        # f1 = f1_score(all_val_y, all_val_pred, average='macro', zero_division=0)
        # cm = confusion_matrix(all_val_y, all_val_pred)
        # print(f"Val Loss in epoch {epoch+1}: {avg_val_loss}, Acc: {acc}, Precision: {precision}, Recall: {recall}, F1: {f1}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict()
            torch.save(best_model_state, f'best_model_{epoch+1}.pth')
            # print(f"Best model updated at epoch {epoch+1}")
        #     plt.figure(figsize=(8, 6))
        #     plt.imshow(cm, cmap='Blues')
        #     plt.colorbar()
        #     plt.xlabel('Predicted')
        #     plt.ylabel('True')
        #     plt.title('Confusion Matrix')
        #     plt.savefig('confusion_matrix.png')
        #     plt.close()
        
    # plt.figure(figsize=(10, 6))
    # plt.plot(range(1, len(training_losses) + 1), training_losses, label='Training Loss', color='blue')
    # plt.plot(range(1, len(val_losses) + 1), val_losses, label='Validation Loss', color='red')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.title('Training and Validation Loss vs. Epoch')
    # plt.legend()
    # plt.savefig('loss_curve.png')
    # plt.close()
    # --- STUDENT CODE: END ---
    
# --- Prediction Function ---
def predict(model, file_path):
    """
    This function makes predictions on a new dataset.

    This function is not used in this file but is provided in case you
    would like to validate the implementation and trained model. A similar
    function will be used by autograder to evaluate your model.
    
    Args:
        model (nn.Module): The trained model.
        file_path (str): Path to the CSV file for which to make predictions.
        
    Returns:
        np.ndarray: A numpy array of predicted labels.
    """
    # Load the data for prediction
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    X = data[:, :-1]
    X_tensor = torch.tensor(X, dtype=torch.float32)
    
    # Set the model to evaluation mode
    model.eval()
    
    # Make and return predictions
    with torch.no_grad():
        outputs = model(X_tensor)
        # Get the index of the max log-probability
        _, predicted_labels = torch.max(outputs.data, 1)
    return predicted_labels.numpy()
    
# --- Main Execution Block ---
if __name__ == '__main__':
    # Instantiate the dataset and dataloader
    train_dataset = CustomDataset('train.csv')
    train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    print("Dataset loaded.")
    
    # Instantiate the model
    model = TwoLayerNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    print("Model initialized.")
    
    # Train the model
    train_model(model, train_loader)
    print("Model training completed.")
    
    # Save the trained model
    torch.save(model.state_dict(), 'model.pth')
    print("Model saved to model.pth")

