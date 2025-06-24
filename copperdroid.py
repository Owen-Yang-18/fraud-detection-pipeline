import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
import pandas as pd
import numpy as np
import os
import random
import optuna
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import cycle

# ===================================================================
#  Data Generation and Graph Construction (Copied for self-containment)
# ===================================================================

def create_dummy_frequency_csv(num_samples=500):
    """
    Generates a dummy CSV file. Increased samples for more robust K-fold validation.
    """
    print(f"Creating a dummy CSV file with {num_samples} samples...")
    syscall_features = ['read', 'write', 'openat', 'execve', 'chmod', 'futex', 'clone', 'mmap', 'close']
    binder_features = ['sendSMS', 'getDeviceId', 'startActivity', 'queryContentProviders', 'getAccounts']
    composite_features = ['NETWORK_WRITE_EXEC', 'READ_CONTACTS(D)', 'DYNAMIC_CODE_LOADING', 'CRYPTO_API_USED']
    features = syscall_features + binder_features + composite_features
    data = np.random.randint(0, 30, size=(num_samples, len(features)))
    df = pd.DataFrame(data, columns=features)
    df.loc[df.sample(frac=0.3).index, np.random.choice(df.columns, 3)] = 0
    labels = []
    for i, row in df.iterrows():
        score = (row['execve']*1.5 + row['sendSMS']*2 + row['NETWORK_WRITE_EXEC']*3 + row['getDeviceId'])
        if score < 40: labels.append(1)
        elif score < 80: labels.append(2)
        elif score < 120: labels.append(3)
        elif score < 160: labels.append(4)
        else: labels.append(5)
    df['Class'] = labels
    file_path = 'app_behavior_frequencies.csv'
    df.to_csv(file_path, index=False)
    print(f"Dummy data saved to '{file_path}'")
    return file_path

def classify_feature_name(name):
    if name.islower(): return 'syscall'
    if not any(c.islower() for c in name): return 'composite'
    return 'binder'

def create_heterogeneous_graph(csv_path):
    """
    Processes a frequency-based CSV into a DGL heterogeneous graph.
    """
    df = pd.read_csv(csv_path)
    app_ids = df.index
    app_map = {name: i for i, name in enumerate(app_ids)}
    action_cols = [col for col in df.columns if col != 'Class']
    syscall_nodes = [col for col in action_cols if classify_feature_name(col) == 'syscall']
    binder_nodes = [col for col in action_cols if classify_feature_name(col) == 'binder']
    composite_nodes = [col for col in action_cols if classify_feature_name(col) == 'composite']
    syscall_map = {name: i for i, name in enumerate(syscall_nodes)}
    binder_map = {name: i for i, name in enumerate(binder_nodes)}
    composite_map = {name: i for i, name in enumerate(composite_nodes)}
    
    app_src, syscall_dst, syscall_freq = [],[],[]
    app_src_b, binder_dst, binder_freq = [],[],[]
    app_src_c, composite_dst, composite_freq = [],[],[]
    for idx in df.index:
        current_app_id = app_map[idx]
        for name in syscall_nodes:
            if df.loc[idx, name] > 0: app_src.append(current_app_id); syscall_dst.append(syscall_map[name]); syscall_freq.append(df.loc[idx, name])
        for name in binder_nodes:
            if df.loc[idx, name] > 0: app_src_b.append(current_app_id); binder_dst.append(binder_map[name]); binder_freq.append(df.loc[idx, name])
        for name in composite_nodes:
            if df.loc[idx, name] > 0: app_src_c.append(current_app_id); composite_dst.append(composite_map[name]); composite_freq.append(df.loc[idx, name])
    
    app_labels = torch.tensor(df['Class'].values - 1, dtype=torch.long)
    graph_data = {
        ('application', 'uses', 'syscall'): (app_src, syscall_dst),
        ('application', 'uses', 'binder'): (app_src_b, binder_dst),
        ('application', 'exhibits', 'composite_behavior'): (app_src_c, composite_dst),
        ('syscall', 'used_by', 'application'): (syscall_dst, app_src),
        ('binder', 'used_by', 'application'): (binder_dst, app_src_b),
        ('composite_behavior', 'exhibited_by', 'application'): (composite_dst, app_src_c)
    }
    g = dgl.heterograph(graph_data, num_nodes_dict={
        'application': len(app_map), 'syscall': len(syscall_map),
        'binder': len(binder_map), 'composite_behavior': len(composite_map)
    })
    g.nodes['application'].data['label'] = app_labels
    
    edge_freqs = {
        ('application', 'uses', 'syscall'): torch.tensor(syscall_freq, dtype=torch.float32),
        ('application', 'uses', 'binder'): torch.tensor(binder_freq, dtype=torch.float32),
        ('application', 'exhibits', 'composite_behavior'): torch.tensor(composite_freq, dtype=torch.float32),
        ('syscall', 'used_by', 'application'): torch.tensor(syscall_freq, dtype=torch.float32),
        ('binder', 'used_by', 'application'): torch.tensor(binder_freq, dtype=torch.float32),
        ('composite_behavior', 'exhibited_by', 'application'): torch.tensor(composite_freq, dtype=torch.float32)
    }
    # Add edge weights
    for etype, data in edge_freqs.items():
        g.edges[etype].data['frequency'] = data
        
    return g

# ===================================================================
#  GNN Model Definition
# ===================================================================

class HeteroGraphSAGE(nn.Module):
    def __init__(self, g, in_size, hidden_size, out_size):
        super().__init__()
        self.embed = dglnn.HeteroEmbedding(g.num_nodes_dict, in_size)
        self.conv1 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(in_size, hidden_size, 'mean', feat_drop=0.2)
            for rel in g.etypes
        }, aggregate='sum')
        self.conv2 = dglnn.HeteroGraphConv({
            rel: dglnn.SAGEConv(hidden_size, hidden_size, 'mean', feat_drop=0.2)
            for rel in g.etypes
        }, aggregate='sum')
        self.classify = nn.Linear(hidden_size, out_size)

    def forward(self, g, inputs=None):
        h = self.conv1(g, self.embed(g.ntypes))
        h = {k: F.relu(v) for k, v in h.items()}
        h = self.conv2(g, h)
        return self.classify(h['application'])

# ===================================================================
#  Training, Evaluation, and Hyperparameter Tuning
# ===================================================================

def evaluate(model, graph, labels, mask, loss_fn):
    """Evaluate model performance on a given dataset."""
    model.eval()
    with torch.no_grad():
        logits = model(graph)
        loss = loss_fn(logits[mask], labels[mask])
        _, indices = torch.max(logits[mask], dim=1)
        accuracy = accuracy_score(labels[mask].cpu(), indices.cpu())
        f1 = f1_score(labels[mask].cpu(), indices.cpu(), average='macro')
        return loss, accuracy, f1, logits[mask], indices

def plot_confusion_matrix(y_true, y_pred, class_names, fold):
    """Plots and saves a confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - Fold {fold+1}')
    plt.ylabel('Actual Class')
    plt.xlabel('Predicted Class')
    filename = f'confusion_matrix_fold_{fold+1}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved confusion matrix to {filename}")

def plot_roc_auc(y_true, y_score, n_classes, fold):
    """Plots and saves a multiclass ROC AUC curve."""
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(10, 8))
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red'])
    for i, color in zip(range(n_classes), colors):
        plt.plot(fpr[i], tpr[i], color=color, lw=2,
                 label='ROC curve of class {0} (area = {1:0.2f})'.format(i, roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Multiclass ROC Curve - Fold {fold+1}')
    plt.legend(loc="lower right")
    filename = f'roc_auc_curve_fold_{fold+1}.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved ROC AUC curve to {filename}")


def objective(trial, g, labels, train_idx, val_idx):
    """Optuna objective function for hyperparameter tuning."""
    embed_size = trial.suggest_categorical('embed_size', [32, 64, 128])
    hidden_size = trial.suggest_categorical('hidden_size', [32, 64, 128])
    lr = trial.suggest_loguniform('lr', 1e-3, 1e-1)
    epochs = 50
    
    num_classes = len(labels.unique())
    model = HeteroGraphSAGE(g, embed_size, hidden_size, num_classes)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        logits = model(g)
        loss = loss_fn(logits[train_idx], labels[train_idx])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    val_loss, val_acc, val_f1, _, _ = evaluate(model, g, labels, val_idx, loss_fn)
    return val_f1

# ===================================================================
#  Main Execution with K-Fold Cross-Validation
# ===================================================================

if __name__ == '__main__':
    print("--- Malware Classification GNN Training ---")
    
    # 1. Prepare Data and Graph
    data_file = create_dummy_frequency_csv()
    graph = create_heterogeneous_graph(data_file)
    
    app_labels = graph.nodes['application'].data['label']
    num_app_nodes = graph.num_nodes('application')
    
    # 2. Setup K-Fold Cross-Validation
    N_SPLITS = 5
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    
    all_fold_metrics = []
    
    print(f"\nStarting {N_SPLITS}-Fold Cross-Validation...")
    
    for fold, (train_idx, test_idx) in enumerate(skf.split(np.arange(num_app_nodes), app_labels)):
        print(f"\n===== FOLD {fold+1}/{N_SPLITS} =====")
        
        train_idx_tensor = torch.tensor(train_idx, dtype=torch.long)
        test_idx_tensor = torch.tensor(test_idx, dtype=torch.long)
        
        # 3. Hyperparameter Tuning with Optuna for this fold
        print("--- Running Hyperparameter Tuning (Optuna) ---")
        # Use a subset of the training data for validation during HPO to speed it up
        sub_train_idx, sub_val_idx = train_test_split(train_idx, test_size=0.2, random_state=42, stratify=app_labels[train_idx])
        
        study = optuna.create_study(direction='maximize', pruner=optuna.pruners.MedianPruner())
        study.optimize(lambda trial: objective(trial, graph, app_labels, sub_train_idx, sub_val_idx), n_trials=20)
        
        best_params = study.best_params
        print(f"Best trial for fold {fold+1}:")
        print(f"  Value (F1 Score): {study.best_value:.4f}")
        print("  Params: ")
        for key, value in best_params.items():
            print(f"    {key}: {value}")

        # 4. Train Final Model for this fold with Best Hyperparameters
        print("\n--- Training Final Model for Fold ---")
        num_classes = len(app_labels.unique())
        class_names = [f'Class {i}' for i in range(num_classes)]
        final_model = HeteroGraphSAGE(graph, best_params['embed_size'], best_params['hidden_size'], num_classes)
        optimizer = torch.optim.Adam(final_model.parameters(), lr=best_params['lr'])
        loss_fn = nn.CrossEntropyLoss()
        
        EPOCHS_FINAL = 100
        for epoch in range(EPOCHS_FINAL):
            final_model.train()
            logits = final_model(graph)
            loss = loss_fn(logits[train_idx_tensor], app_labels[train_idx_tensor])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch+1:03d}/{EPOCHS_FINAL} | Loss: {loss.item():.4f}")

        # 5. Evaluate on the Test Set for this fold
        test_loss, test_acc, test_f1, test_logits, test_preds = evaluate(final_model, graph, app_labels, test_idx_tensor, loss_fn)
        print(f"\n--- Evaluation for Fold {fold+1} ---")
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test F1-Score (Macro): {test_f1:.4f}")
        
        all_fold_metrics.append({'acc': test_acc, 'f1': test_f1})

        # 6. Generate and save plots for this fold
        print("\n--- Generating Evaluation Plots ---")
        plot_confusion_matrix(app_labels[test_idx_tensor].cpu(), test_preds.cpu(), class_names, fold)
        
        y_score = F.softmax(test_logits, dim=1)
        plot_roc_auc(app_labels[test_idx_tensor].cpu(), y_score.cpu(), num_classes, fold)

    # 7. Report Final Averaged Results
    print("\n\n===== FINAL CROSS-VALIDATION RESULTS =====")
    avg_acc = np.mean([m['acc'] for m in all_fold_metrics])
    std_acc = np.std([m['acc'] for m in all_fold_metrics])
    avg_f1 = np.mean([m['f1'] for m in all_fold_metrics])
    std_f1 = np.std([m['f1'] for m in all_fold_metrics])

    print(f"Average Accuracy over {N_SPLITS} folds: {avg_acc:.4f} (+/- {std_acc:.4f})")
    print(f"Average F1-Score over {N_SPLITS} folds: {avg_f1:.4f} (+/- {std_f1:.4f})")

    # Cleanup
    os.remove(data_file)
    print("\nCleanup complete. Check for 'confusion_matrix_fold_*.png' and 'roc_auc_curve_fold_*.png' files.")
