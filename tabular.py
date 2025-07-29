import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, multilabel_confusion_matrix, ConfusionMatrixDisplay
)
import optuna
import matplotlib.pyplot as plt

# ─── Data loading & preprocessing ───
def load_main_data(path):
    df = pd.read_csv(path)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1].astype(int).values - 1
    return X, y

def select_top_k_features(X, imp_csv, k):
    imp = pd.read_csv(imp_csv).sort_values('importance', ascending=False)
    feats = [f for f in imp['feature'].tolist() if f in X.columns][:k]
    return X[feats], feats

def preprocess(csv_main, csv_imp, top_k):
    X_all, y = load_main_data(csv_main)
    X_sel, feat_names = select_top_k_features(X_all, csv_imp, top_k)
    scaler = StandardScaler()
    X_norm = scaler.fit_transform(X_sel)
    return X_norm.astype(np.float32), y, feat_names

# ─── MLP model ───
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, n_layers, dropout, num_classes):
        super().__init__()
        layers = []
        h = input_dim
        for _ in range(n_layers):
            layers += [nn.Linear(h, hidden_dim), nn.ReLU(), nn.Dropout(dropout)]
            h = hidden_dim
        layers.append(nn.Linear(h, num_classes))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# ─── Optuna objective ───
def objective(trial, X, y, input_dim, num_classes,
              n_splits=5, batch_size=64, epochs=20):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    hidden_dim = trial.suggest_int("hidden_dim", 32, 256)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-2)

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0)
    val_accs = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    for train_idx, val_idx in skf.split(X, y):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        dl_train = DataLoader(TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train)),
                              batch_size=batch_size, shuffle=True)
        dl_val = DataLoader(TensorDataset(torch.from_numpy(X_val), torch.from_numpy(y_val)),
                            batch_size=batch_size, shuffle=False)

        model = MLP(input_dim, hidden_dim, n_layers, dropout, num_classes).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        model.train()
        for _ in range(epochs):
            for xb, yb in dl_train:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad()
                loss = criterion(model(xb), yb)
                loss.backward()
                opt.step()

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for xb, yb in dl_val:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb).argmax(dim=1)
                correct += int((preds == yb).sum())
                total += yb.size(0)
        val_accs.append(correct/total)

    return float(np.mean(val_accs))

# ─── Evaluate best model on test split, compute metrics, plot & save confusion matrix ───
def evaluate_best(X, y, best_params, test_size=0.2, batch_size=64, epochs=20, cm_path="confusion_matrix.png"):
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=0)
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = MLP(input_dim, best_params['hidden_dim'],
                best_params['n_layers'], best_params['dropout'], num_classes).to(device)
    opt = torch.optim.AdamW(model.parameters(),
                             lr=best_params['lr'],
                             weight_decay=best_params['weight_decay'])
    criterion = nn.CrossEntropyLoss()

    dl_tr = DataLoader(TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr)),
                       batch_size=batch_size, shuffle=True)
    dl_te = DataLoader(TensorDataset(torch.from_numpy(X_te), torch.from_numpy(y_te)),
                       batch_size=batch_size, shuffle=False)

    model.train()
    for _ in range(epochs):
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in dl_te:
            xb, yb = xb.to(device), yb.to(device)
            y_pred.append(model(xb).argmax(dim=1).cpu().numpy())
            y_true.append(yb.cpu().numpy())
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)

    # Basic metrics
    acc = accuracy_score(y_true, y_pred)
    f1m = f1_score(y_true, y_pred, average='macro', zero_division=0)
    prec_m = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec_m = recall_score(y_true, y_pred, average='macro', zero_division=0)

    print("Test accuracy:", acc)
    print("Macro F1:", f1m, "Precision:", prec_m, "Recall:", rec_m)

    # Compute per-class confusion matrices
    mlcm = multilabel_confusion_matrix(y_true, y_pred, labels=range(num_classes))
    tprs = []; tnrs = []; fprs = []; fnrs = []
    for cm in mlcm:
        tn, fp, fn, tp = cm.ravel()
        tprs.append(tp / (tp + fn) if tp + fn else 0.0)
        tnrs.append(tn / (tn + fp) if tn + fp else 0.0)
        fprs.append(fp / (fp + tn) if fp + tn else 0.0)
        fnrs.append(fn / (fn + tp) if fn + tp else 0.0)

    print(f"Avg TPR: {np.mean(tprs):.4f}, Avg TNR: {np.mean(tnrs):.4f}, "
          f"Avg FPR: {np.mean(fprs):.4f}, Avg FNR: {np.mean(fnrs):.4f}")

    print("\nPer-class confusion matrices (TN FP / FN TP):")
    for cls, cm in enumerate(mlcm):
        print(f"Class {cls}:")
        print(cm)

    print("\nFull confusion matrix:")
    full_cm = confusion_matrix(y_true, y_pred)
    print(full_cm)

    # Plot normalized confusion matrix
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, labels=range(num_classes),
        normalize='true', cmap=plt.cm.Blues
    )
    disp.ax_.set_title("Normalized Confusion Matrix")
    fig = disp.figure_
    fig.savefig(cm_path)
    plt.close(fig)
    print(f"Saved confusion matrix plot to {cm_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_main", type=str, required=True)
    parser.add_argument("--csv_imp", type=str, required=True)
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--cm_path", type=str, default="confusion_matrix.png")
    args = parser.parse_args()

    X, y, feats = preprocess(args.csv_main, args.csv_imp, args.top_k)
    print("Selected features:", feats)
    input_dim = X.shape[1]
    num_classes = len(np.unique(y))

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(lambda t: objective(t, X, y, input_dim, num_classes, n_splits=args.folds),
                   n_trials=args.trials)

    print("Best validation accuracy:", study.best_value)
    print("Best hyperparameters:", study.best_params)

    evaluate_best(X, y, study.best_params, test_size=args.test_size, cm_path=args.cm_path)

if __name__ == "__main__":
    main()
