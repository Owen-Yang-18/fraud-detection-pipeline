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





import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import optuna
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score, 
    confusion_matrix
)
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

class PyTorchMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout_rate=0.1):
        super(PyTorchMLP, self).__init__()
        
        for i in range(1, len(hidden_dims)):
            if hidden_dims[i] > hidden_dims[i-1]:
                raise ValueError(f"Hidden dimensions must be non-increasing. "
                               f"Layer {i}: {hidden_dims[i]} > Layer {i-1}: {hidden_dims[i-1]}")
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

def calculate_multiclass_metrics(y_true, y_pred, num_classes):
    cm = confusion_matrix(y_true, y_pred)
    
    tpr = np.zeros(num_classes)
    tnr = np.zeros(num_classes) 
    fpr = np.zeros(num_classes)
    fnr = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        tpr[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        tnr[i] = tn / (tn + fp) if (tn + fp) > 0 else 0
        fpr[i] = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr[i] = fn / (tp + fn) if (tp + fn) > 0 else 0
    
    return tpr, tnr, fpr, fnr

def train_pytorch_model(model, train_loader, val_loader, num_epochs, learning_rate, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    model.train()
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
    
    model.eval()
    val_predictions = []
    val_targets = []
    
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs.data, 1)
            val_predictions.extend(predicted.cpu().numpy())
            val_targets.extend(batch_y.cpu().numpy())
    
    return accuracy_score(val_targets, val_predictions)

def pytorch_objective(trial, X, y, num_classes, device, cv_folds=5):
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4])
    hidden_dims = []
    
    prev_dim = X.shape[1]
    for i in range(num_layers):
        max_dim = min(512, prev_dim) if i == 0 else prev_dim
        choices = [32, 64, 128, 256, 512]
        valid_choices = [c for c in choices if c <= max_dim]
        hidden_dim = trial.suggest_categorical(f'hidden_dim_{i}', valid_choices)
        hidden_dims.append(hidden_dim)
        prev_dim = hidden_dim
    
    dropout_rate = trial.suggest_categorical('dropout_rate', [0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    num_epochs = trial.suggest_categorical('num_epochs', [10, 20, 30, 40, 50])
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create data loaders
        train_dataset = TensorDataset(
            torch.FloatTensor(X_train_fold), 
            torch.LongTensor(y_train_fold)
        )
        val_dataset = TensorDataset(
            torch.FloatTensor(X_val_fold), 
            torch.LongTensor(y_val_fold)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Create and train model
        model = PyTorchMLP(X.shape[1], hidden_dims, num_classes, dropout_rate).to(device)
        accuracy = train_pytorch_model(model, train_loader, val_loader, num_epochs, learning_rate, device)
        scores.append(accuracy)
    
    return np.mean(scores)

def sklearn_objective(trial, X, y, cv_folds=5):
    num_layers = trial.suggest_categorical('num_layers', [1, 2, 3, 4])
    hidden_layer_sizes = []
    
    prev_dim = X.shape[1]
    for i in range(num_layers):
        max_dim = min(512, prev_dim) if i == 0 else prev_dim
        choices = [32, 64, 128, 256, 512]
        valid_choices = [c for c in choices if c <= max_dim]
        hidden_dim = trial.suggest_categorical(f'hidden_dim_{i}', valid_choices)
        hidden_layer_sizes.append(hidden_dim)
        prev_dim = hidden_dim
    
    learning_rate = trial.suggest_float('learning_rate_init', 1e-5, 1e-2, log=True)
    alpha = trial.suggest_float('alpha', 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])
    max_iter = trial.suggest_categorical('max_iter', [100, 200, 300, 400, 500])
    
    # Cross-validation
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    scores = []
    
    for train_idx, val_idx in skf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        
        # Create and train model
        model = MLPClassifier(
            hidden_layer_sizes=tuple(hidden_layer_sizes),
            learning_rate_init=learning_rate,
            alpha=alpha,
            batch_size=batch_size,
            max_iter=max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        model.fit(X_train_fold, y_train_fold)
        accuracy = model.score(X_val_fold, y_val_fold)
        scores.append(accuracy)
    
    return np.mean(scores)

def evaluate_model_cv(model, X, y, num_classes, model_type='sklearn', cv_folds=5, **model_kwargs):
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    
    fold_metrics = {
        'accuracy': [], 'f1_macro': [], 'recall_macro': [], 'precision_macro': [],
        'tpr': [], 'tnr': [], 'fpr': [], 'fnr': []
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        print(f"Evaluating fold {fold + 1}/{cv_folds}...")
        
        X_train_fold, X_test_fold = X[train_idx], X[test_idx]
        y_train_fold, y_test_fold = y[train_idx], y[test_idx]
        
        if model_type == 'pytorch':
            pytorch_model = PyTorchMLP(
                X.shape[1], 
                model_kwargs['hidden_dims'], 
                num_classes, 
                model_kwargs.get('dropout_rate', 0.1)
            ).to(device)
            
            train_dataset = TensorDataset(
                torch.FloatTensor(X_train_fold), 
                torch.LongTensor(y_train_fold)
            )
            test_dataset = TensorDataset(
                torch.FloatTensor(X_test_fold), 
                torch.LongTensor(y_test_fold)
            )
            
            train_loader = DataLoader(train_dataset, batch_size=model_kwargs.get('batch_size', 64), shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=model_kwargs.get('batch_size', 64), shuffle=False)
            
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(pytorch_model.parameters(), lr=model_kwargs.get('learning_rate', 0.001))
            
            pytorch_model.train()
            for epoch in range(model_kwargs.get('num_epochs', 50)):
                for batch_x, batch_y in train_loader:
                    batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                    
                    optimizer.zero_grad()
                    outputs = pytorch_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    optimizer.step()
            
            pytorch_model.eval()
            y_pred = []
            with torch.no_grad():
                for batch_x, batch_y in test_loader:
                    batch_x = batch_x.to(device)
                    outputs = pytorch_model(batch_x)
                    _, predicted = torch.max(outputs.data, 1)
                    y_pred.extend(predicted.cpu().numpy())
            
        else:
            sklearn_model = MLPClassifier(**model_kwargs, random_state=42)
            sklearn_model.fit(X_train_fold, y_train_fold)
            y_pred = sklearn_model.predict(X_test_fold)
        
        accuracy = accuracy_score(y_test_fold, y_pred)
        f1_macro = f1_score(y_test_fold, y_pred, average='macro')
        recall_macro = recall_score(y_test_fold, y_pred, average='macro')
        precision_macro = precision_score(y_test_fold, y_pred, average='macro')
        
        tpr, tnr, fpr, fnr = calculate_multiclass_metrics(y_test_fold, y_pred, num_classes)
        
        fold_metrics['accuracy'].append(accuracy)
        fold_metrics['f1_macro'].append(f1_macro)
        fold_metrics['recall_macro'].append(recall_macro)
        fold_metrics['precision_macro'].append(precision_macro)
        fold_metrics['tpr'].append(np.mean(tpr))
        fold_metrics['tnr'].append(np.mean(tnr))
        fold_metrics['fpr'].append(np.mean(fpr))
        fold_metrics['fnr'].append(np.mean(fnr))
        
        print(f"Fold {fold + 1} - Accuracy: {accuracy:.4f}, F1-Macro: {f1_macro:.4f}")
    
    avg_metrics = {}
    for metric, values in fold_metrics.items():
        avg_metrics[f'{metric}_mean'] = np.mean(values)
        avg_metrics[f'{metric}_std'] = np.std(values)
    
    return avg_metrics, fold_metrics

def main():
    print("=== ML Pipeline with Feature Selection and Hyperparameter Tuning ===\n")
    
    print("1. Loading data...")
    try:
        data_df = pd.read_csv('your_data_file.csv')
        ranking_df = pd.read_csv('your_ranking_file.csv')
        
        print(f"Data shape: {data_df.shape}")
        print(f"Ranking shape: {ranking_df.shape}")
    except FileNotFoundError:
        print("Error: Please make sure your CSV files exist and update the file paths in the script.")
        print("Expected files:")
        print("- your_data_file.csv (with 'Class' column)")
        print("- your_ranking_file.csv (with feature rankings)")
        return
    
    print("\n2. Preparing features and target...")
    X = data_df.drop('Class', axis=1)
    y = data_df['Class']
    
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    num_classes = len(np.unique(y_encoded))
    
    print(f"Number of classes: {num_classes}")
    print(f"Class distribution: {np.bincount(y_encoded)}")
    
    print("\n3. Selecting top 64 features...")
    ranking_columns = ranking_df.columns.tolist()
    print(f"Ranking file columns: {ranking_columns}")
    
    top_features = ranking_df.nsmallest(64, ranking_columns[1])[ranking_columns[0]].tolist()
    
    available_features = [f for f in top_features if f in X.columns]
    if len(available_features) < 64:
        print(f"Warning: Only {len(available_features)} features available from ranking file")
        available_features = X.columns.tolist()[:64]
    else:
        available_features = available_features[:64]
    
    X_selected = X[available_features]
    print(f"Selected {X_selected.shape[1]} features")
    
    print("\n4. Preprocessing with MinMaxScaler...")
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print("\n5. Hyperparameter tuning for PyTorch MLP...")
    pytorch_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    pytorch_study.optimize(
        lambda trial: pytorch_objective(trial, X_scaled, y_encoded, num_classes, device),
        n_trials=50
    )
    
    print("Best PyTorch hyperparameters:")
    for key, value in pytorch_study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best PyTorch CV score: {pytorch_study.best_value:.4f}")
    
    print("\n6. Hyperparameter tuning for sklearn MLP...")
    sklearn_study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler())
    sklearn_study.optimize(
        lambda trial: sklearn_objective(trial, X_scaled, y_encoded),
        n_trials=50
    )
    
    print("Best sklearn hyperparameters:")
    for key, value in sklearn_study.best_params.items():
        print(f"  {key}: {value}")
    print(f"Best sklearn CV score: {sklearn_study.best_value:.4f}")
    
    print("\n7. Final evaluation with 5-fold CV...")
    
    pytorch_best = pytorch_study.best_params
    pytorch_hidden_dims = []
    for i in range(pytorch_best['num_layers']):
        pytorch_hidden_dims.append(pytorch_best[f'hidden_dim_{i}'])
    
    print(f"PyTorch architecture: {X_scaled.shape[1]} -> {' -> '.join(map(str, pytorch_hidden_dims))} -> {num_classes}")
    
    pytorch_kwargs = {
        'hidden_dims': pytorch_hidden_dims,
        'dropout_rate': pytorch_best['dropout_rate'],
        'learning_rate': pytorch_best['learning_rate'],
        'batch_size': pytorch_best['batch_size'],
        'num_epochs': pytorch_best['num_epochs']
    }
    
    sklearn_best = sklearn_study.best_params
    sklearn_hidden_dims = []
    for i in range(sklearn_best['num_layers']):
        sklearn_hidden_dims.append(sklearn_best[f'hidden_dim_{i}'])
    
    print(f"Sklearn architecture: {X_scaled.shape[1]} -> {' -> '.join(map(str, sklearn_hidden_dims))} -> {num_classes}")
    
    sklearn_kwargs = {
        'hidden_layer_sizes': tuple(sklearn_hidden_dims),
        'learning_rate_init': sklearn_best['learning_rate_init'],
        'alpha': sklearn_best['alpha'],
        'batch_size': sklearn_best['batch_size'],
        'max_iter': sklearn_best['max_iter'],
        'early_stopping': True,
        'validation_fraction': 0.1
    }
    
    print("\n--- PyTorch MLP Results ---")
    pytorch_avg_metrics, pytorch_fold_metrics = evaluate_model_cv(
        None, X_scaled, y_encoded, num_classes, 'pytorch', **pytorch_kwargs
    )
    
    print("\n--- Sklearn MLP Results ---")
    sklearn_avg_metrics, sklearn_fold_metrics = evaluate_model_cv(
        None, X_scaled, y_encoded, num_classes, 'sklearn', **sklearn_kwargs
    )
    
    print("\n" + "="*80)
    print("FINAL RESULTS SUMMARY")
    print("="*80)
    
    metrics_names = ['accuracy', 'f1_macro', 'recall_macro', 'precision_macro', 'tpr', 'tnr', 'fpr', 'fnr']
    
    print("\nPyTorch MLP Results:")
    print("-" * 50)
    for metric in metrics_names:
        mean_val = pytorch_avg_metrics[f'{metric}_mean']
        std_val = pytorch_avg_metrics[f'{metric}_std']
        print(f"{metric.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nSklearn MLP Results:")
    print("-" * 50)
    for metric in metrics_names:
        mean_val = sklearn_avg_metrics[f'{metric}_mean']
        std_val = sklearn_avg_metrics[f'{metric}_std']
        print(f"{metric.upper():15s}: {mean_val:.4f} ± {std_val:.4f}")
    
    print("\nFold-wise Results:")
    print("-" * 50)
    print("PyTorch MLP - Fold-wise Accuracy:", [f"{acc:.4f}" for acc in pytorch_fold_metrics['accuracy']])
    print("Sklearn MLP - Fold-wise Accuracy:", [f"{acc:.4f}" for acc in sklearn_fold_metrics['accuracy']])
    
    pytorch_avg_acc = pytorch_avg_metrics['accuracy_mean']
    sklearn_avg_acc = sklearn_avg_metrics['accuracy_mean']
    
    print(f"\nModel Comparison:")
    print("-" * 50)
    if pytorch_avg_acc > sklearn_avg_acc:
        print(f"PyTorch MLP performs better: {pytorch_avg_acc:.4f} vs {sklearn_avg_acc:.4f}")
    else:
        print(f"Sklearn MLP performs better: {sklearn_avg_acc:.4f} vs {pytorch_avg_acc:.4f}")
    
    print("\nPipeline completed successfully!")

if __name__ == "__main__":
    main()