import pandas as pd
import numpy as np
import dgl
import torch
import os
import random
from pyvis.network import Network

def create_dummy_frequency_csv(num_samples=100):
    """
    Generates a dummy CSV file that matches your description.
    The 'Class' column now contains multi-class labels from 1 to 5.
    Increased sample size to ensure enough samples per class for visualization.
    """
    print(f"Creating a dummy CSV file with {num_samples} samples...")
    
    # Define feature names that follow the user's rules
    syscall_features = ['read', 'write', 'openat', 'execve', 'chmod', 'futex']
    binder_features = ['sendSMS', 'getDeviceId', 'startActivity', 'queryContentProviders']
    composite_features = ['NETWORK_WRITE_EXEC', 'READ_CONTACTS(D)', 'DYNAMIC_CODE_LOADING']
    
    features = syscall_features + binder_features + composite_features
    
    # Create random frequency data
    data = np.random.randint(0, 30, size=(num_samples, len(features)))
    df = pd.DataFrame(data, columns=features)
    
    # Make some data zero to ensure not all nodes are connected
    df.loc[df.sample(frac=0.3).index, np.random.choice(df.columns, 3)] = 0
    
    # Generate multi-class labels (1-5) based on a score
    labels = []
    for i, row in df.iterrows():
        malicious_score = (
            row['execve'] * 1.5 + row['sendSMS'] * 2 + row['NETWORK_WRITE_EXEC'] * 3
        )
        if malicious_score < 40:
            labels.append(1) # Benign
        elif malicious_score < 80:
            labels.append(2) # Potentially Unwanted App (PUA)
        elif malicious_score < 120:
            labels.append(3) # Adware
        elif malicious_score < 160:
            labels.append(4) # Spyware
        else:
            labels.append(5) # Ransomware/Trojan
            
    df['Class'] = labels # The label column is now named 'Class'
    
    file_path = 'app_behavior_frequencies.csv'
    df.to_csv(file_path, index=False)
    print(f"Dummy data saved to '{file_path}'")
    return file_path

def classify_feature_name(name):
    """Classifies a column name as syscall, binder, or composite."""
    if name.islower():
        return 'syscall'
    if not any(c.islower() for c in name):
        return 'composite'
    return 'binder'

def create_heterogeneous_graph(csv_path):
    """
    Processes a frequency-based CSV into a DGL heterogeneous graph and
    returns the graph, DataFrame, and lists of action nodes.
    """
    print(f"\n--- Starting DGL Graph Construction from {csv_path} ---")

    # 1. Load Data
    print("Step 1: Loading data...")
    df = pd.read_csv(csv_path)

    # 2. Identify Node Types and Create Mappings
    app_ids = df.index
    app_map = {name: i for i, name in enumerate(app_ids)}
    action_cols = [col for col in df.columns if col != 'Class']
    syscall_nodes = [col for col in action_cols if classify_feature_name(col) == 'syscall']
    binder_nodes = [col for col in action_cols if classify_feature_name(col) == 'binder']
    composite_nodes = [col for col in action_cols if classify_feature_name(col) == 'composite']
    syscall_map = {name: i for i, name in enumerate(syscall_nodes)}
    binder_map = {name: i for i, name in enumerate(binder_nodes)}
    composite_map = {name: i for i, name in enumerate(composite_nodes)}
    
    # 3. Prepare Edge Lists and Edge Features
    app_src, syscall_dst, syscall_freq = [], [], []
    app_src_b, binder_dst, binder_freq = [], [], []
    app_src_c, composite_dst, composite_freq = [], [], []
    for idx in df.index:
        current_app_id = app_map[idx]
        for action_name in syscall_nodes:
            frequency = df.loc[idx, action_name]
            if frequency > 0:
                app_src.append(current_app_id); syscall_dst.append(syscall_map[action_name]); syscall_freq.append(frequency)
        for action_name in binder_nodes:
            frequency = df.loc[idx, action_name]
            if frequency > 0:
                app_src_b.append(current_app_id); binder_dst.append(binder_map[action_name]); binder_freq.append(frequency)
        for action_name in composite_nodes:
            frequency = df.loc[idx, action_name]
            if frequency > 0:
                app_src_c.append(current_app_id); composite_dst.append(composite_map[action_name]); composite_freq.append(frequency)

    # 4. Prepare Node Labels
    app_labels = torch.tensor(df['Class'].values - 1, dtype=torch.long)

    # 5. Construct the Graph
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

    # 6. Add Data to Graph
    g.nodes['application'].data['label'] = app_labels
    g.edges[('application', 'uses', 'syscall')].data['frequency'] = torch.tensor(syscall_freq, dtype=torch.float32)
    g.edges[('application', 'uses', 'binder')].data['frequency'] = torch.tensor(binder_freq, dtype=torch.float32)
    g.edges[('application', 'exhibits', 'composite_behavior')].data['frequency'] = torch.tensor(composite_freq, dtype=torch.float32)
    g.edges[('syscall', 'used_by', 'application')].data['frequency'] = torch.tensor(syscall_freq, dtype=torch.float32)
    g.edges[('binder', 'used_by', 'application')].data['frequency'] = torch.tensor(binder_freq, dtype=torch.float32)
    g.edges[('composite_behavior', 'exhibited_by', 'application')].data['frequency'] = torch.tensor(composite_freq, dtype=torch.float32)

    print("--- DGL Graph construction complete! ---")
    return g, df, syscall_nodes, binder_nodes, composite_nodes

def visualize_heterogeneous_graph(df, all_syscalls, all_binders, all_composites):
    """
    Builds an interactive pyvis visualization from a pre-loaded DataFrame and
    pre-computed lists of action nodes, now with reciprocal edges.
    """
    print(f"\n--- Starting Pyvis Visualization ---")

    # 1. Use the passed DataFrame directly
    print("Step 1: Using pre-loaded data...")

    # 2. Randomly sample subsets from the provided action lists
    print("Step 2: Subsetting data for visualization...")
    
    syscall_subset = random.sample(all_syscalls, k=min(len(all_syscalls), 3))
    binder_subset = random.sample(all_binders, k=min(len(all_binders), 2))
    composite_subset = random.sample(all_composites, k=min(len(all_composites), 2))
    action_subset = syscall_subset + binder_subset + composite_subset
    
    app_subset_df = df.groupby('Class').head(10)

    # 3. Setup Pyvis Network and Color Mappings
    print("Step 3: Initializing Pyvis network...")
    net = Network(height="750px", width="100%", bgcolor="#222222", font_color="white", notebook=True, directed=True)
    
    node_color_map = {
        'application': '#ff69b4', # Pink
        'syscall': '#6495ed',     # Cornflower Blue
        'binder': '#ff8c00',      # Dark Orange
        'composite': '#c71585'    # Medium Violet Red
    }
    edge_color_map = {
        'uses_syscall': '#6495ed',
        'uses_binder': '#ff8c00',
        'uses_composite': '#c71585'
    }

    # 4. Add Nodes to the Pyvis Graph
    print("Step 4: Adding nodes to the graph...")
    # Add application nodes
    for idx, row in app_subset_df.iterrows():
        app_class = int(row['Class'])
        net.add_node(f"app_{idx}", 
                     label=f"App {idx}", 
                     title=f"Class: {app_class}",
                     color=node_color_map['application'],
                     size=25)

    # Add action nodes
    for action in action_subset:
        action_type = classify_feature_name(action)
        node_id = f"action_{action}"
        net.add_node(node_id, label=action, title=f"Type: {action_type}", color=node_color_map[action_type], size=15)

    # 5. Add Edges to the Pyvis Graph
    print("Step 5: Adding reciprocal edges to the graph...")
    for idx, row in app_subset_df.iterrows():
        for action_name in action_subset:
            if action_name in row and row[action_name] > 0:
                frequency = int(row[action_name])
                action_type = classify_feature_name(action_name)
                source_node = f"app_{idx}"
                dest_node = f"action_{action_name}"
                edge_color_key = f"uses_{action_type.replace('_behavior', '')}"
                edge_color = edge_color_map.get(edge_color_key)
                
                # Add the forward edge (app -> action)
                net.add_edge(source_node, dest_node, title=f"Frequency: {frequency}", 
                             value=frequency, color=edge_color)
                
                # REFACTOR: Add the reciprocal edge (action -> app)
                net.add_edge(dest_node, source_node, title=f"Frequency: {frequency}", 
                             value=frequency, color=edge_color)
    
    # 6. Generate the visualization
    output_filename = "heterogeneous_graph_visualization.html"
    print(f"Step 6: Saving visualization to '{output_filename}'...")
    net.show(output_filename)
    print("--- Pyvis Visualization complete! ---")


if __name__ == '__main__':
    data_file_path = create_dummy_frequency_csv()
    
    # 1. Create the DGL graph and get the dataframe and action lists in return
    dgl_graph, loaded_df, syscalls, binders, composites = create_heterogeneous_graph(data_file_path)
    if dgl_graph:
        print("\n--- DGL Graph Summary ---")
        print(dgl_graph)
    
    # 2. Create the Pyvis visualization by passing the loaded data
    visualize_heterogeneous_graph(loaded_df, syscalls, binders, composites)

    # --- Cleanup ---
    os.remove(data_file_path)
    print(f"\nCleaned up dummy CSV file. Please open 'heterogeneous_graph_visualization.html' to see the graph.")
