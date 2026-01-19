# #!/usr/bin/env python3
# """
# create_snapshots_fromCSV_weighted.py

# Usage:
# python create_snapshots_fromCSV_weighted.py --input visit_matrix.csv --weights weights.csv --history 7 --train_size 70
# """
# import argparse
# import numpy as np
# import pandas as pd

# def create_snapshots(data: np.ndarray, history: int):
#     num_nodes, num_days = data.shape
#     if num_days < 740:
#         raise ValueError(f"Data must have at least 740 days, got {num_days}")
#     num_snapshots = 740 - history
#     X = np.stack([data[:, t:t+history] for t in range(num_snapshots)], axis=0)
#     y = np.stack([data[:, t+history] for t in range(num_snapshots)], axis=0)
#     return X, y

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--input", required=True, help="CSV file path with visits")
#     parser.add_argument("--weights", required=True, help="CSV file path with edge weights")
#     parser.add_argument("--history", type=int, default=7, help="history window")
#     parser.add_argument("--train_size", type=int, default=70, help="train percent")
#     args = parser.parse_args()

#     # Load visit matrix
#     df = pd.read_csv(args.input)
#     if 'tile_id' in df.columns:
#         df = df.drop(columns=['tile_id'])
#     else:
#         first_col = df.columns[0]
#         if not np.issubdtype(df[first_col].dtype, np.number):
#             df = df.drop(columns=[first_col])
#     df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
#     data = df.to_numpy(dtype=float)
#     num_nodes, num_days = data.shape
#     print(f"Loaded data: {num_nodes} nodes × {num_days} days")

#     # Load weight matrix
#     weights_df = pd.read_csv(args.weights, header=None)
#     weights = weights_df.to_numpy(dtype=float)
#     if weights.shape != (num_nodes, num_nodes):
#         raise ValueError(f"Weight matrix shape {weights.shape} does not match number of nodes {num_nodes}")
#     print(f"Loaded weight matrix: {weights.shape}")

#     # Create snapshots
#     X, y = create_snapshots(data, args.history)
#     total = X.shape[0]
#     split_idx = int(total * args.train_size / 100)
#     X_train, y_train = X[:split_idx], y[:split_idx]
#     X_test, y_test = X[split_idx:], y[split_idx:]

#     print(f"Total snapshots: {total}, train: {len(X_train)}, test: {len(X_test)}")
#     np.savez("graph_snapshots_weighted.npz",
#              X_train=X_train, y_train=y_train,
#              X_test=X_test, y_test=y_test,
#              weights=weights)
#     print("\nSaved graph_snapshots_weighted.npz")

# if __name__ == "__main__":
#     main()


#!/usr/bin/env python3
"""
create_snapshots_fromCSV_weighted.py

Chronological snapshot creation with weights
Train/Val/Test = 60/20/20
Context size T ∈ {8,16}
Horizon = 1
"""

import argparse
import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

def create_snapshots(data: np.ndarray, history: int, horizon: int = 1):
    num_nodes, num_days = data.shape

    num_snapshots = num_days - history - horizon + 1
    if num_snapshots <= 0:
        raise ValueError("Not enough days for given history + horizon")

    X = np.stack(
        [data[:, t:t+history] for t in range(num_snapshots)],
        axis=0
    )  # (S, N, T)

    y = np.stack(
        [data[:, t+history+horizon-1] for t in range(num_snapshots)],
        axis=0
    )  # (S, N)

    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--history", type=int, choices=[8, 16], required=True)
    args = parser.parse_args()

    # -------------------------
    # Load visit matrix
    # -------------------------
    df = pd.read_csv(args.input)
    if 'tile_id' in df.columns:
        df = df.drop(columns=['tile_id'])
    else:
        first_col = df.columns[0]
        if not np.issubdtype(df[first_col].dtype, np.number):
            df = df.drop(columns=[first_col])

    df = df.apply(pd.to_numeric, errors='coerce').fillna(0.0)
    data = df.to_numpy(dtype=float)
    num_nodes, num_days = data.shape

    print(f"Visits: {num_nodes} nodes × {num_days} days")

    # -------------------------
    # Load weight matrix
    # -------------------------
    weights = pd.read_csv(args.weights, header=None).to_numpy(dtype=float)
    if weights.shape != (num_nodes, num_nodes):
        raise ValueError("Weight matrix shape mismatch")

    # -------------------------
    # Create snapshots
    # -------------------------
    X, y = create_snapshots(data, args.history, horizon=1)
    total = X.shape[0]

    train_end = int(0.6 * total)
    val_end   = int(0.8 * total)

    X_train, y_train = X[:train_end], y[:train_end]
    X_val,   y_val   = X[train_end:val_end], y[train_end:val_end]
    X_test,  y_test  = X[val_end:], y[val_end:]

    print(f"Snapshots → train {len(X_train)}, val {len(X_val)}, test {len(X_test)}")

    np.savez(
        "graph_snapshots_weighted.npz",
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        X_test=X_test,   y_test=y_test,
        weights=weights
    )

    print("Saved graph_snapshots_weighted.npz")

if __name__ == "__main__":
    main()
