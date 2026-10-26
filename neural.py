"""A small, explicitly designed neural network for tabular transactions."""
import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import average_precision_score

class FraudNet(nn.Module):
    def __init__(self, n_features=30, hidden=(64, 32), dropout=.2):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_features, hidden[0]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(hidden[1], 1))
    def forward(self, x):
        return self.layers(x).squeeze(1)  # raw logits, not probabilities

@torch.no_grad()
def predict_nn(model, x):
    model.eval()  # disables dropout for repeatable inference
    chunks = []
    for start in range(0, len(x), 4096):
        batch = torch.tensor(np.asarray(x[start:start+4096]), dtype=torch.float32)
        chunks.append(torch.sigmoid(model(batch)).numpy())
    return np.concatenate(chunks)

def train_nn(x_train, y_train, x_val, y_val, hidden, epochs=20, seed=42):
    torch.manual_seed(seed)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    model = FraudNet(x_train.shape[1], hidden)
    y = np.asarray(y_train, dtype=np.float32)
    # Fraud is rare: increase its contribution to the training loss.
    weight = float((y == 0).sum() / (y == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weight))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001, weight_decay=1e-4)
    dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y))
    loader = DataLoader(dataset, batch_size=512, shuffle=True,
                        generator=torch.Generator().manual_seed(seed), num_workers=0)
    best_ap, stale, best_state, history = -1., 0, None, []
    for epoch in range(1, epochs+1):
        model.train()
        total = 0.
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item()*len(xb)
        ap = float(average_precision_score(y_val, predict_nn(model, x_val)))
        history.append({'epoch': epoch, 'train_loss': total/len(y), 'validation_ap': ap})
        print(f'NN {hidden}, epoch {epoch}: validation AP={ap:.4f}', flush=True)
        if ap > best_ap + 1e-6:
            best_ap, stale = ap, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= 4:
            break
    model.load_state_dict(best_state)
    return model, history, weight
