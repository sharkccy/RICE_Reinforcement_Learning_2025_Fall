import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from oracle import Oracle

# --- Constants ---
# NOTE: Do not change these.
ORACLE_BUDGET = 100     # Maximum number of oracle queries allowed
RANDOM_STATE = 442      # For reproducibility
INPUT_SIZE = 32         # Number of input features
OUTPUT_SIZE = 5         # Number of output classes
EPOCHS = 100            # Training epochs for neural network



# --- Dataset Definition ---
class CustomDataset(Dataset):
    """A custom dataset class for loading the training data."""
    def __init__(self, X, y=None):
        """
        Initialize dataset with features and optional labels.
        
        Args:
            X (np.ndarray): Feature data
            y (np.ndarray, optional): Label data
        """
        self.X = torch.tensor(X, dtype=torch.float32)
        if y is not None:
            self.y = torch.tensor(y, dtype=torch.long)
        else:
            self.y = None
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        else:
            return self.X[idx]

# --- Neural Network Definition ---
class TwoLayerNN(nn.Module):
    """A simple two-layer neural network for classification."""
    def __init__(self, input_size, hidden_size, output_size):
        """
        Initializes the TwoLayerNN model.

        Args:
            input_size (int): The size of the input features.
            hidden_size (int): The number of neurons in the hidden layer.
            output_size (int): The number of output classes.
        """
        super(TwoLayerNN, self).__init__()
        # --- SUBMISSION CODE: START ---
        # TODO: Define the neural network architecture
        self.layer = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_size),
        )
        
        # --- SUBMISSION CODE: END ---
    
    def forward(self, x):
        """
        Defines the forward pass of the neural network.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output of the network.
        """        
        # --- SUBMISSION CODE: START ---
        # TODO: Implement the forward pass
        
        return self.layer(x)
        
        # --- SUBMISSION CODE: END ---

# DO NOT MODIFY THIS CLASS SIGNATURE. THE GRADER WILL USE THE SAME CLASS SIGNATURE.
class ActiveLearner:
    """
    Active learning class that implements a strategy for selecting 
    the most informative samples to label.
    """
    
    def __init__(self, random_state=RANDOM_STATE):
        """
        Initialize the active learner.
        
        Args:
            random_state (int): Random state for reproducibility.
        """
        self.random_state = random_state
        torch.manual_seed(random_state)
        np.random.seed(random_state)
        
        self.model = None
        self.labeled_X = None
        self.labeled_y = None
        
        # --- SUBMISSION CODE: START ---
        
        
        # --- SUBMISSION CODE: END ---
    
    def train_initial_model(self, X_supervised, y_supervised):
        """
        Train the initial model on the supervised dataset.
        
        Args:
            X_supervised (np.ndarray): Supervised features.
            y_supervised (np.ndarray): Supervised labels.
        """
        # --- SUBMISSION CODE: START ---
        # TODO: Train your initial neural network on the supervised data
        init_lr = 5e-2
        lr_decay_factor = 0.95
        weight_decay = 1e-5
        batch_size = 64

        self.model = TwoLayerNN(INPUT_SIZE, 64, OUTPUT_SIZE)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)

        loss = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=init_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay_factor)
        scaler = torch.amp.GradScaler()

        train_dataset = CustomDataset(X_supervised, y_supervised)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        self.labeled_X = X_supervised
        self.labeled_y = y_supervised

        training_losses = []

        for epoch in range(EPOCHS):
            self.model.train()
            epoch_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type):
                    outputs = self.model(inputs)
                    loss_value = loss(outputs, labels)
                
                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()                                     #adjust the scale parameter for next iteration
                epoch_loss += loss_value.item() * inputs.size(0)
            
            scheduler.step()
            training_losses.append(epoch_loss / len(train_dataset))

            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {training_losses[-1]:.4f}")

        # --- SUBMISSION CODE: END ---
    
    def select_samples(self, X_unsupervised, n_samples=1):
        """
        Select the most informative samples from the unsupervised pool.
        
        This is the core of your active learning strategy!
        
        Args:
            X_unsupervised (np.ndarray): Unlabeled features to choose from.
            n_samples (int): Number of samples to select.
            
        Returns:
            list: Indices of selected samples in X_unsupervised.
        """
        # --- SUBMISSION CODE: START ---
        # TODO: Implement your active learning strategy

        # Empty / None input or non-positive request
        if X_unsupervised is None or len(X_unsupervised) == 0 or n_samples <= 0:
            return []

        # Safety: ensure we have a trained model
        if self.model is None:
            return []

        # Normalize input to numpy array for consistent indexing
        X_unsupervised = np.asarray(X_unsupervised)

        # Make sure n_samples does not exceed pool size
        n_samples = min(n_samples, len(X_unsupervised))



        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        dataset = CustomDataset(X_unsupervised)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        prob_list = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                prob_list.append(probs.cpu())
        
        all_probs = torch.cat(prob_list, dim=0).cpu()
        eps = 1e-10
        entropy = -torch.sum(all_probs * torch.log(all_probs + eps), dim=1)      # entropy = -∑(p*log(p))
        selected_indices = torch.topk(entropy, n_samples).indices.cpu().numpy().tolist()

        return selected_indices

        # --- SUBMISSION CODE: END ---
    
    def update_model(self, X_new, y_new, fine_tune_epochs=10):
        """
        Update the model with newly labeled samples.
        
        Args:
            X_new (np.ndarray): New features to add to training set.
            y_new (np.ndarray): New labels to add to training set.
        """
        # --- SUBMISSION CODE: START ---
        # TODO: Add new samples to your training set and retrain the neural network

        # nothing to add
        if X_new is None:
            return []

        # Normalize to numpy arrays
        X_new = np.asarray(X_new)
        y_new = np.asarray(y_new)

        # Ensure shapes: make X_new 2D and y_new 1D
        if X_new.ndim == 1:
            X_new = X_new.reshape(1, -1)
        y_new = y_new.reshape(-1)

        # Check counts
        if X_new.shape[0] != y_new.shape[0]:
            raise ValueError("X_new and y_new must have the same number of samples (first dim).")

        # If no labeled data yet, initialize; otherwise append
        if self.labeled_X is None or len(self.labeled_X) == 0:
            # copy to avoid accidental views
            self.labeled_X = X_new.copy()
            self.labeled_y = y_new.copy()
        else:
            # Ensure existing labeled_X is numpy array
            self.labeled_X = np.asarray(self.labeled_X)
            self.labeled_y = np.asarray(self.labeled_y)
            # vstack/concatenate (preserve dtype)
            self.labeled_X = np.vstack([self.labeled_X, X_new])
            self.labeled_y = np.concatenate([self.labeled_y, y_new])

        # Delegate actual training to the helper
        losses = self._train_neural_network(epochs=fine_tune_epochs,
                                            lr=51e-3,
                                            batch_size=min(64, max(1, len(self.labeled_X))),
                                            weight_decay=1e-5,
                                            lr_decay=0.95,
                                            print_progress=True)

        return losses
        
        # --- SUBMISSION CODE: END ---
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Features to predict.
            
        Returns:
            np.ndarray: Predicted labels.
        """
        # --- SUBMISSION CODE: START ---
        # TODO: Return predictions from your trained neural network

        if self.model is None:
            raise ValueError("Model is not trained yet.")
        self.model.eval()
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device)
        dataset = CustomDataset(X)
        dataloader = DataLoader(dataset, batch_size=64, shuffle=False)
        all_preds = []
        with torch.no_grad():
            for inputs in dataloader:
                inputs = inputs.to(device)
                outputs = self.model(inputs)
                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())

        all_preds = torch.cat(all_preds, dim=0).numpy()
        return all_preds
        # Placeholder: how you would return predictions if you had a model
        return np.random.choice(OUTPUT_SIZE, len(X))
        
        # --- SUBMISSION CODE: END --
    
    def _train_neural_network(self, epochs=10, lr=1e-3, batch_size=32, weight_decay=1e-5, lr_decay=0.95, print_progress=True):
        """
        Helper method to train the neural network on current labeled data.
        """
        # --- SUBMISSION CODE: START ---
        # TODO: Implement neural network training loop
        
        if self.labeled_X is None or self.labeled_y is None or len(self.labeled_X) == 0:
            if print_progress:
                print("[_train_neural_network] No labeled data to train on.")
            return []

        # Ensure numpy arrays and shapes
        # X = np.asarray(self.labeled_X)
        # y = np.asarray(self.labeled_y).reshape(-1)
        # if X.ndim == 1:
        #     X = X.reshape(1, -1)
        # if X.shape[0] != y.shape[0]:
        #     raise ValueError("self.labeled_X and self.labeled_y must have the same number of samples.")

        # Ensure model exists
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.model is None:
            self.model = TwoLayerNN(INPUT_SIZE, 64, OUTPUT_SIZE)
            if print_progress:
                print("[_train_neural_network] Model created.")
            self.model.to(device)
        else:
            self.model.to(device)

        # Prepare DataLoader
        train_dataset = CustomDataset(self.labeled_X, self.labeled_y)
        trainloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Loss / optimizer / scaler / scheduler
        loss_fn = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=lr_decay)
        scaler = torch.amp.GradScaler()

        training_losses = []
        self.model.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for inputs, labels in trainloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                with torch.amp.autocast(device_type=device.type):
                    outputs = self.model(inputs)
                    loss_value = loss_fn(outputs, labels)

                scaler.scale(loss_value).backward()
                scaler.step(optimizer)
                scaler.update()
                epoch_loss += loss_value.item() * inputs.size(0)

            scheduler.step()
            avg_loss = epoch_loss / len(train_dataset)
            training_losses.append(avg_loss)
            if print_progress:
                print(f"[_train_neural_network] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        # keep model on device (already moved)
        return training_losses
        # --- SUBMISSION CODE: END ---


def load_data(file_path, has_labels=True):
    """
    Load data from CSV file.
    
    Args:
        file_path (str): Path to CSV file.
        has_labels (bool): Whether the file contains labels.
        
    Returns:
        tuple: (X, y) if has_labels=True, else (X, None)
    """
    data = pd.read_csv(file_path)
    
    if has_labels:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values
        return X, y
    else:
        X = data.values
        return X, None

# DO NOT MODIFY THIS FUNCTION SIGNATURE. THE GRADER WILL USE THE SAME FUNCTION SIGNATURE.
def active_learning_loop(learner, X_unsupervised, oracle, budget=ORACLE_BUDGET):
    """
    Main active learning loop.
    
    Args:
        learner (ActiveLearner): The active learning instance.
        X_unsupervised (np.ndarray): Unlabeled data pool.
        oracle (Oracle): Oracle that provides labels for selected samples.
        budget (int): Maximum number of oracle queries.
        
    Returns:
        ActiveLearner: The trained learner after active learning.
    """
    # --- SUBMISSION CODE: START ---
    # TODO: Implement the main active learning loop
    # 
    # Suggested structure:
    # 1. Select informative samples using learner.select_samples()
    # 2. Query the oracle for labels: oracle.get_label(sample_index)
    # 3. Update the model with new labeled samples
    # 4. Remove labeled samples from the unsupervised pool
    # 5. Repeat until budget is exhausted

        # basic guards
    if X_unsupervised is None or len(X_unsupervised) == 0 or budget <= 0:
        return learner

    X_unsupervised = np.asarray(X_unsupervised)
    pool_idx = np.arange(len(X_unsupervised))              # global indices of remaining pool
    remaining_budget = int(budget)

    # how many queries per round (tuneable)
    per_round = 10

    while remaining_budget > 0 and pool_idx.size > 0:
        # current pool features (no heavy deletion; just index into original)
        X_pool = X_unsupervised[pool_idx]

        # decide how many to ask this round
        k = min(per_round, remaining_budget, len(X_pool))
        if k <= 0:
            break

        # get local indices from learner (relative to X_pool)
        local_idxs = learner.select_samples(X_pool, k)
        if not local_idxs:
            # nothing selected -> stop
            break

        # ensure numpy array of ints and within bounds
        local_idxs = np.asarray(local_idxs, dtype=int)
        # Clip/unique to be safe (avoid duplicates or out-of-range)
        local_idxs = np.unique(local_idxs)
        local_idxs = local_idxs[local_idxs >= 0]
        local_idxs = local_idxs[local_idxs < len(X_pool)]
        if local_idxs.size == 0:
            break

        # map to global indices to query oracle
        global_idxs = pool_idx[local_idxs]

        # Query oracle for each global index (collect labels)
        X_sel = X_pool[local_idxs]
        y_sel = []
        for gi in global_idxs:
            # oracle.get_label may reduce its internal budget; here we also track remaining_budget
            label = oracle.get_label(int(gi))
            y_sel.append(label)
            remaining_budget -= 1
            if remaining_budget <= 0:
                # if budget exhausted by oracle or internal tracking, stop querying further
                # (we still keep the labels we've collected so far)
                pass

        y_sel = np.asarray(y_sel)

        # Update model with newly labeled samples (append+fine-tune)
        learner.update_model(X_sel, y_sel, fine_tune_epochs=30)

        # Remove selected samples from pool_idx using boolean mask (vectorized, safe)
        mask = np.ones(pool_idx.shape[0], dtype=bool)
        mask[local_idxs] = False
        pool_idx = pool_idx[mask]

        # loop continues until budget exhausted or pool empty
    

    # Placeholder: do nothing (implement your loop!)
    # print(f"Active learning loop not implemented. Budget: {budget}")
    
    # --- SUBMISSION CODE: END ---
    
    return learner


# --- Main Execution Block ---
if __name__ == '__main__':
    print("=== NEURAL NETWORK ACTIVE LEARNING ===")
    print("Loading datasets...")
    
    # Load supervised and unsupervised data
    X_supervised, y_supervised = load_data('train_supervised.csv', has_labels=True)
    X_unsupervised, _ = load_data('train_unsupervised.csv', has_labels=False)
    
    print(f"Supervised data: {X_supervised.shape[0]} samples, {X_supervised.shape[1]} features")
    print(f"Unsupervised data: {X_unsupervised.shape[0]} samples")
    print(f"Classes: {OUTPUT_SIZE}")
    
    # Initialize active learner
    learner = ActiveLearner(random_state=RANDOM_STATE)
    
    # Train initial model
    print("Training initial neural network...")
    learner.train_initial_model(X_supervised, y_supervised)
    
    # Test initial accuracy 
    try:
        X_test, y_test = load_data('test.csv', has_labels=True)
        print("\n=== INITIAL MODEL PERFORMANCE ===")
        initial_predictions = learner.predict(X_test)
        initial_accuracy = accuracy_score(y_test, initial_predictions)
        print(f"INITIAL test accuracy: {initial_accuracy:.4f}")
    except FileNotFoundError:
        print("Test data not available for initial evaluation")
    
    # Try to initialize oracle for local testing
    try:
        oracle = Oracle()
        
        # Run active learning loop if oracle is available
        if oracle._labels:  # Check if labels were loaded
            print("\nRunning active learning loop...")
            trained_learner = active_learning_loop(learner, X_unsupervised, oracle, budget=ORACLE_BUDGET)
            
            # Test final accuracy
            try:
                print("\n=== FINAL MODEL PERFORMANCE ===")
                final_predictions = trained_learner.predict(X_test)
                final_accuracy = accuracy_score(y_test, final_predictions)
                print(f"FINAL test accuracy: {final_accuracy:.4f}")
                print(f"IMPROVEMENT: {final_accuracy - initial_accuracy:.4f} ({((final_accuracy - initial_accuracy) / initial_accuracy * 100):.1f}%)")
                
                # Show oracle statistics
                stats = oracle.get_statistics()
                print(f"\nOracle Usage Statistics:")
                print(f"  - Queries used: {stats['queries_used']}/{stats['total_budget']}")
                print(f"  - Budget utilization: {stats['budget_utilization']:.2%}")
                print(f"  - Unique samples queried: {stats['unique_samples_queried']}")
                
            except NameError:
                print("Test data not available for final evaluation")
        else:
            print("Oracle labels not available - cannot run active learning loop")
            print("This is expected when running starter code without oracle labels")
            
    except Exception as e:
        print(f"Oracle initialization failed: {e}")
        print("This is expected when running starter code without oracle labels")
    
    print("\nNeural Network Active Learning setup complete!")
    print("Remember to implement all TODO sections in your code!")
    print("The grader will provide the oracle and evaluate your active learning strategy.")