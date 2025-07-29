import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
import optuna

# ─── 1. Load main CSV ───────────────────────────────────────────────
def load_main_data(csv_path):
    df = pd.read_csv(csv_path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int).values - 1  # labels 1–5 → 0–4
    return X, y

# ─── 2. Select top‑k features via importance CSV ────────────────
def select_top_k_features(X: pd.DataFrame, imp_csv: str, k: int):
    imp = pd.read_csv(imp_csv)
    imp = imp.sort_values('importance', ascending=False)
    feats = [f for f in imp['feature'].tolist() if f in X.columns][:k]
    return X[feats], feats

# ─── 3. Preprocess: feature selection + normalization ───────────────
def preprocess(csv_main, csv_imp, top_k):
    X_all, y = load_main_data(csv_main)
    X_sel, feat_names = select_top_k_features(X_all, csv_imp, top_k)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_sel)
    return X_norm.astype(np.float32), y, feat_names

# ─── 4. Define MLP model ─────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, num_classes):
        super().__init__()
        layers = []
        h = input_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(h, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            h = hidden_dim
        layers.append(nn.Linear(h, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ─── 5. Objective fn with Stratified CV & Optuna tuning ──────────────
def objective(trial, X, y, input_dim, num_classes, n_splits=5,
              batch_size=64, epochs=20):
    # Hyperparameters to tune
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    val_accuracies = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        train_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
            batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
            batch_size=batch_size, shuffle=False)

        model = MLP(input_dim, hidden_dim, n_layers, dropout, num_classes).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for xb, yb in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += int((preds == yb).sum())
                total += yb.size(0)
        val_accuracies.append(correct / total)

    return float(np.mean(val_accuracies))

# ─── 6. Main function ─────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_main", type=str, required=True)
    parser.add_argument("--csv_imp", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50)
    args = parser.parse_args()

    X, y, feats = preprocess(args.csv_main, args.csv_imp, args.top_k)
    print("Selected features:", feats)
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, X, y, input_dim, num_classes, n_splits=args.folds),
        n_trials=args.trials
    )

    print("Best validation accuracy: {:.4f}".format(study.best_value))
    print("Best hyperparameters:", study.best_params)
