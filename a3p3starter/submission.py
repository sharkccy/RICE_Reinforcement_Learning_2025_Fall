import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score

# --------------------------
# Global, Reproducible Seeding
# --------------------------
RANDOM_STATE = 442


def set_seed(seed: int = RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass
    torch.set_num_threads(1)


set_seed(RANDOM_STATE)

# --------------------------
# Constants
# --------------------------
ORACLE_BUDGET = 128
BATCH_SIZE = 16
INPUT_SIZE = 32
HIDDEN_SIZE = 128
OUTPUT_SIZE = 5
EPOCHS = 100
LEARNING_RATE = 1e-3


# --------------------------
# Dataset
# --------------------------
class CustomDataset(Dataset):

    def __init__(self, X, y=None):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y is None:
            return self.X[idx]
        return self.X[idx], self.y[idx]


# --------------------------
# Model
# --------------------------
class TwoLayerNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_size // 2, output_size),
        )

    def forward(self, x):
        return self.network(x)


# --------------------------
# Active Learner
# --------------------------
class ActiveLearner:
    """
    Uncertainty sampling via entropy over softmax probabilities.
    """

    def __init__(self, random_state=RANDOM_STATE):
        set_seed(random_state)
        self.random_state = random_state
        self.model = TwoLayerNN(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        self.labeled_X = None
        self.labeled_y = None

    def train_initial_model(self, X_supervised, y_supervised):
        self.labeled_X = X_supervised.copy()
        self.labeled_y = y_supervised.copy()
        self._train_neural_network(is_initial=True)

    def select_samples(self, X_unsupervised, n_samples=1):
        if self.model is None or len(X_unsupervised) == 0:
            if len(X_unsupervised) == 0:
                return []
            rng = np.random.default_rng(self.random_state)
            return rng.choice(len(X_unsupervised),
                              min(n_samples, len(X_unsupervised)),
                              replace=False).tolist()

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X_unsupervised, dtype=torch.float32)
            logits = self.model(X_tensor)

            temperature = 1.5
            probs = torch.softmax(logits / temperature, dim=1)

            eps = 1e-12
            probs = torch.clamp(probs, eps, 1 - eps)
            entropy = (-probs * torch.log(probs)).sum(dim=1)  # [N]

            # Deterministic tie-breaker: sort by (entropy desc, index asc)
            ent_np = entropy.cpu().numpy()
            idx = np.arange(len(ent_np))
            # lexsort sorts by last key fastest; we want entropy descending -> use -ent
            order = np.lexsort((idx, -ent_np))
            n = min(n_samples, len(X_unsupervised))
            return order[:n].tolist()

    def update_model(self, X_new, y_new):
        if self.labeled_X is None:
            self.labeled_X = X_new.copy()
            self.labeled_y = y_new.copy()
        else:
            self.labeled_X = np.vstack([self.labeled_X, X_new])
            self.labeled_y = np.concatenate([self.labeled_y, y_new])
        self._train_neural_network(is_initial=False)

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32)
            out = self.model(X_tensor)
            _, pred = torch.max(out, 1)
            return pred.numpy()

    def _train_neural_network(self, is_initial=False):
        if self.labeled_X is None or len(self.labeled_X) == 0:
            return
        dataset = CustomDataset(self.labeled_X, self.labeled_y)
        gen = torch.Generator()
        gen.manual_seed(self.random_state)
        loader = DataLoader(dataset,
                            batch_size=BATCH_SIZE,
                            shuffle=True,
                            generator=gen,
                            num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=LEARNING_RATE,
                                     weight_decay=1e-4)
        epochs = EPOCHS if is_initial else 15

        self.model.train()
        for _ in range(epochs):
            for Xb, yb in loader:
                optimizer.zero_grad()
                logits = self.model(Xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()


# --------------------------
# Oracle that uses the learner's model directly (to implement)
# --------------------------
class SelfModelOracle:
    """
    An oracle that uses an externally provided model (the learner's model)
    to produce labels. Enforces a fixed query budget.
    """

    def __init__(self, budget=ORACLE_BUDGET, seed=RANDOM_STATE):
        """
        TODO: Initialize this class.
        """
        set_seed(seed)
        # --- SUBMISSION CODE: START ---
        self.budget = budget
        self.queries_made = 0
        self.model = None
        # --- SUBMISSION CODE: END ---

    def attach_model(self, model: nn.Module):
        """
        TODO: Attach a trained/being-trained model (ActiveLearner.model).
        """
        # --- SUBMISSION CODE: START ---
        self.model = model
        # --- SUBMISSION CODE: END ---

    def get_label(self, sample: np.ndarray) -> int:
        """
        TODO: return the model's argmax class for a single sample.
        Must:
          - enforce budget,
          - use the attached model in eval mode,
          - be deterministic given a fixed model.
        """
        # --- SUBMISSION CODE: START ---
        self.model.eval()
        if self.queries_made >= self.budget:
            return None  # budget exhausted
        self.queries_made += 1
        with torch.no_grad():
            sample_tensor = torch.tensor(sample, dtype=torch.float32).unsqueeze(0)
            logits = self.model(sample_tensor)
            return logits.argmax().item()
        
        # --- SUBMISSION CODE: END ---


# --------------------------
# Data IO
# --------------------------
def load_data(file_path, has_labels=True):
    data = pd.read_csv(file_path)
    if has_labels:
        X = data.iloc[:, :-1].values
        y = data.iloc[:, -1].values.astype(int)
        return X, y
    else:
        return data.values, None


# --------------------------
# Active learning loop
# --------------------------
def active_learning_loop(learner: ActiveLearner,
                         X_unsupervised: np.ndarray,
                         oracle: SelfModelOracle,
                         budget: int = ORACLE_BUDGET,
                         batch_size: int = 5):
    # Keep a moving set of available indices for pool-based AL
    available = list(range(len(X_unsupervised)))
    queries = 0
    round_id = 0

    while queries < budget and len(available) > 0:
        cur_budget = min(batch_size, budget - queries, len(available))
        if cur_budget == 0:
            break

        pool = X_unsupervised[available]
        rel_idx = learner.select_samples(pool, n_samples=cur_budget)
        abs_idx = [available[i] for i in rel_idx]

        add_X, add_y = [], []
        for i in abs_idx:
            x = X_unsupervised[i:i + 1]
            y_hat = oracle.get_label(
                x[0])  # oracle labels one sample at a time
            if y_hat:
                add_X.append(x)
                add_y.append(y_hat)

        if add_X:
            batch_X = np.vstack(add_X)
            batch_y = np.array(add_y, dtype=int)
            learner.update_model(batch_X, batch_y)

        for i in abs_idx:
            available.remove(i)
        queries += len(abs_idx)
        round_id += 1

    return learner


# --------------------------
# Main
# --------------------------
if __name__ == "__main__":
    print("=== A3P3: Self-Oracle Active Learning (STARTER) ===")

    X_sup, y_sup = load_data('train_supervised.csv', has_labels=True)
    X_unsup, _ = load_data('train_unsupervised.csv', has_labels=False)
    X_test, y_test = load_data('test.csv', has_labels=True)

    learner = ActiveLearner(random_state=RANDOM_STATE)
    learner.train_initial_model(X_sup, y_sup)

    initial_acc = accuracy_score(y_test, learner.predict(X_test))
    print(f"Initial test accuracy: {initial_acc:.4f}")

    oracle = SelfModelOracle(budget=ORACLE_BUDGET, seed=RANDOM_STATE)
    oracle.attach_model(learner.model)
    learner = active_learning_loop(learner,
                                   X_unsup,
                                   oracle,
                                   budget=ORACLE_BUDGET,
                                   batch_size=BATCH_SIZE // 4)

    final_acc = accuracy_score(y_test, learner.predict(X_test))
    print(f"Final test accuracy:   {final_acc:.4f}")
    print(f"Improvement:           {final_acc - initial_acc:+.4f}")
