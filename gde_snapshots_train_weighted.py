#!/usr/bin/env python3
"""
Fast GCDE training with weighted edges, validation early stopping, and optional CSV saving
"""

import argparse, math, random, numpy as np, torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from torchdiffeq import odeint
import pandas as pd

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------
# GCDE Function
# -------------------------
class GCDEFunc(torch.nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.gcn1 = GCNConv(dim, hidden)                   # 2 GCN's
        self.gcn2 = GCNConv(hidden, dim)
        self.act = torch.nn.ReLU()

    def forward(self, t, x):
        x = self.act(self.gcn1(x, self.edge_index, self.edge_weight))      
        x = self.gcn2(x, self.edge_index, self.edge_weight)
        return x

# -------------------------
# GCDE Model
# -------------------------
class GCDERegressor(torch.nn.Module):
    def __init__(self, dim, hidden, edge_index, edge_weight):
        super().__init__()
        self.func = GCDEFunc(dim, hidden)
        self.func.edge_index = edge_index
        self.func.edge_weight = edge_weight
        self.mlp = torch.nn.Linear(dim, 1)                        # MLP head

    def forward(self, x0, solver="rk4"):
        t = torch.tensor([0.0, 1.0], device=x0.device)
        xt = odeint(self.func, x0, t, method=solver)
        return self.mlp(xt[-1]).squeeze(-1)

# -------------------------
# Utils
# -------------------------
def build_edge_index_weights(weights):
    weights = np.array(weights, dtype=np.float32)
    src, dst = np.nonzero(weights)
    edge_index = torch.tensor(np.stack([src, dst]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(weights[src, dst], dtype=torch.float32, device=device)
    return edge_index.contiguous(), edge_weight

def build_graphs(X, y, edge_index, edge_weight):
    X = torch.tensor(np.array(X, dtype=np.float32), device=device)
    y = torch.tensor(np.array(y, dtype=np.float32), device=device)
    return [Data(x=Xi, y=yi, edge_index=edge_index, edge_weight=edge_weight) for Xi, yi in zip(X, y)]

def mae(pred, target):
    return torch.mean(torch.abs(pred - target)).item()

def rmse(pred, target):
    return math.sqrt(torch.mean((pred - target) ** 2).item())

# -------------------------
# Training function
# -------------------------
def train(npz_file, hidden, lr, solver="rk4", patience=250, max_epochs=500, batch_size=8, save_csv=None):
    data = np.load(npz_file)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val, y_val = data["X_val"], data["y_val"]
    X_test, y_test = data["X_test"], data["y_test"]
    weights = data["weights"]

    N, T = X_train.shape[1], X_train.shape[2]
    edge_index, edge_weight = build_edge_index_weights(weights)

    train_loader = DataLoader(build_graphs(X_train, y_train, edge_index, edge_weight),
                              batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(build_graphs(X_val, y_val, edge_index, edge_weight),
                            batch_size=batch_size)
    test_loader = DataLoader(build_graphs(X_test, y_test, edge_index, edge_weight),
                             batch_size=batch_size)

    model = GCDERegressor(T, hidden, edge_index, edge_weight).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()

    best_val_mae = float("inf")
    best_state = None
    patience_ctr = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        # Train
        model.train()
        train_mse = 0.0
        for batch in train_loader:
            opt.zero_grad()
            loss = loss_fn(model(batch.x, solver), batch.y)
            loss.backward()
            opt.step()
            train_mse += loss.item()
        train_mse /= len(train_loader)

        # Validation
        model.eval()
        val_mae = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_mae += mae(model(batch.x, solver), batch.y)
        val_mae /= len(val_loader)

        # Early stopping
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_state = model.state_dict()
            patience_ctr = 0
        else:
            patience_ctr += 1

        # Save history
        history.append({
            "epoch": epoch,
            "train_mse": train_mse,
            "val_mae": val_mae,
            "best_val_mae_so_far": best_val_mae
        })

        if patience_ctr >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

        if epoch % 50 == 0:
            print(f"Epoch {epoch} | Train MSE {train_mse:.4f} | Val MAE {val_mae:.4f}")

    # Save history CSV if requested
    if save_csv:
        pd.DataFrame(history).to_csv(save_csv, index=False)

    # Test
    model.load_state_dict(best_state)           # <-- load the best checkpoint based on validation
    model.eval()
    test_mae, test_rmse = 0.0, 0.0
    with torch.no_grad():
        for batch in test_loader:
            pred = model(batch.x, solver)
            test_mae += mae(pred, batch.y)
            test_rmse += rmse(pred, batch.y)
    test_mae /= len(test_loader)
    test_rmse /= len(test_loader)

    print(f"TEST | MAE={test_mae:.4f}, RMSE={test_rmse:.4f}")

    return {"MAE": test_mae, "RMSE": test_rmse, "history": history}

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--npz", required=True)
    parser.add_argument("--hidden", type=int, choices=[16, 24], required=True)
    parser.add_argument("--lr", type=float, choices=[0.01, 0.001], required=True)
    parser.add_argument("--solver", default="rk4")
    parser.add_argument("--save_csv", type=str, default=None,
                        help="Optional path to save train/val history CSV")
    args = parser.parse_args()
    train(args.npz, args.hidden, args.lr, args.solver, save_csv=args.save_csv)
