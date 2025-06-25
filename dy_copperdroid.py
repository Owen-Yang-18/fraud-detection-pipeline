import os
import pickle
import random
import time
import pandas as pd
import math

# ===================================================================
#  Step 1: Data Simulation (Replace with your actual data)
# ===================================================================

def create_dummy_pkl_files(directory="apk_pkl_logs", num_files=50):
    """
    Generates a directory of dummy .pkl files, each simulating a processed
    APK log. NO LONGER creates a labels.csv file.
    """
    print(f"Creating {num_files} dummy .pkl files in '{directory}'...")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(num_files):
        apk_id = f"apk_{i}"
        filename = os.path.join(directory, f"{apk_id}.pkl")
        
        event_list = []
        current_time = time.time() + i # Stagger start times
        
        for _ in range(random.randint(20, 100)): # Vary number of events
            current_time += random.uniform(0.001, 0.5)
            action_type = random.choice(['syscall', 'binder', 'composite'])
            if action_type == 'syscall':
                action_name = random.choice(['read', 'write', 'openat', 'execve'])
            elif action_type == 'binder':
                action_name = random.choice(['sendSMS', 'getDeviceId', 'startActivity'])
            else:
                action_name = random.choice(['NETWORK_WRITE_EXEC', 'FS_ACCESS(WRITE)'])
            
            event_list.append((current_time, action_name))
        
        # Save the event list for this APK
        with open(filename, 'wb') as f:
            pickle.dump(event_list, f)
        
    print("Dummy data creation complete.")


# ===================================================================
#  Step 2: Fusion and Chunked Graph Data Preparation
# ===================================================================

def classify_action_name(name):
    """Classifies an action name as syscall, binder, or composite."""
    if name.islower():
        return 'syscall'
    # Simple check: if it contains non-alphanumeric chars (besides _) or is all caps
    if not name.islower() and not name.isupper():
        return 'binder'
    return 'composite_behavior'

def save_graph_data_in_chunks(graph_data, output_directory, chunk_size=1000000):
    """
    Saves the graph data to disk, splitting large edge lists into smaller chunks.
    
    Args:
        graph_data (dict): Dictionary containing node maps and edge lists.
        output_directory (str): The directory to save the output files.
        chunk_size (int): The maximum number of edges to store in a single chunk file.
    """
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # 1. Save the node mappings (these are small)
    node_maps = {
        'apk_map': graph_data['apk_map'],
        'syscall_map': graph_data['syscall_map'],
        'binder_map': graph_data['binder_map'],
        'composite_map': graph_data['composite_map']
    }
    with open(os.path.join(output_directory, 'node_maps.pkl'), 'wb') as f:
        pickle.dump(node_maps, f)
    print(f"Saved node mappings to '{os.path.join(output_directory, 'node_maps.pkl')}'")

    # 2. Save the edge lists in chunks
    edge_types = ['app_syscall', 'app_binder', 'app_composite']
    for edge_type in edge_types:
        edge_list = graph_data[edge_type]
        num_chunks = math.ceil(len(edge_list) / chunk_size)
        print(f"Saving '{edge_type}' edges in {num_chunks} chunk(s)...")

        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size
            chunk_data = edge_list[start:end]
            
            chunk_filename = f"{edge_type}_edges_chunk_{i}.pkl"
            with open(os.path.join(output_directory, chunk_filename), 'wb') as f:
                pickle.dump(chunk_data, f)
    
    print(f"All chunked graph data has been saved to '{output_directory}'")


def prepare_graph_data(data_directory, output_directory="graph_data_chunks"):
    """
    Fuses all .pkl files into a unified event stream and saves the resulting
    graph data (node maps, chunked edges) to disk.
    """
    print("\n--- Starting Unified Graph Data Preparation ---")

    # 1. Scan for all pkl files
    print(f"Step 1: Scanning for .pkl files in '{data_directory}'...")
    pkl_files = [f for f in os.listdir(data_directory) if f.endswith('.pkl')]
    if not pkl_files:
        print("Error: No .pkl files found in the directory.")
        return
        
    # 2. Fuse all events into a master list
    print(f"Step 2: Fusing {len(pkl_files)} PKL files into a single event stream...")
    master_event_list = []
    for pkl_file in pkl_files:
        apk_id = os.path.splitext(pkl_file)[0]
        filepath = os.path.join(data_directory, pkl_file)
        try:
            with open(filepath, 'rb') as f:
                events = pickle.load(f)
                for timestamp, action_name in events:
                    master_event_list.append((timestamp, apk_id, action_name))
        except Exception as e:
            print(f"Warning: Could not process file {pkl_file}. Error: {e}")
    
    # 3. Sort the master list by timestamp
    print("Step 3: Sorting the unified event stream chronologically...")
    master_event_list.sort(key=lambda x: x[0])

    # 4. Identify all unique nodes and create mappings
    print("Step 4: Identifying all unique nodes...")
    all_apks = sorted(list({os.path.splitext(f)[0] for f in pkl_files}))
    all_actions = set(event[2] for event in master_event_list)
    
    apk_map = {name: i for i, name in enumerate(all_apks)}
    
    syscall_nodes = {name for name in all_actions if classify_action_name(name) == 'syscall'}
    binder_nodes = {name for name in all_actions if classify_action_name(name) == 'binder'}
    composite_nodes = {name for name in all_actions if classify_action_name(name) == 'composite_behavior'}

    syscall_map = {name: i for i, name in enumerate(sorted(list(syscall_nodes)))}
    binder_map = {name: i for i, name in enumerate(sorted(list(binder_nodes)))}
    composite_map = {name: i for i, name in enumerate(sorted(list(composite_nodes)))}

    print(f"- Found {len(apk_map)} unique application nodes.")
    print(f"- Found {len(syscall_map)} unique syscall nodes.")
    print(f"- Found {len(binder_map)} unique binder nodes.")
    print(f"- Found {len(composite_map)} unique composite_behavior nodes.")

    # 5. Build edge lists from the event stream
    print("Step 5: Building edge lists and timestamp features...")
    edge_data = {
        'app_syscall': [], 'app_binder': [], 'app_composite': []
    }
    
    for timestamp, apk_id, action_name in master_event_list:
        apk_idx = apk_map[apk_id]
        action_type = classify_action_name(action_name)
        
        if action_type == 'syscall':
            action_idx = syscall_map[action_name]
            edge_data['app_syscall'].append((apk_idx, action_idx, timestamp))
        elif action_type == 'binder':
            action_idx = binder_map[action_name]
            edge_data['app_binder'].append((apk_idx, action_idx, timestamp))
        elif action_type == 'composite_behavior':
            action_idx = composite_map[action_name]
            edge_data['app_composite'].append((apk_idx, action_idx, timestamp))

    # 6. Explicitly sort each edge list by timestamp
    print("Step 6: Verifying timestamp sort order for all edge types...")
    for edge_type, edge_list in edge_data.items():
        # This sort is a safeguard; the order should already be correct
        # from iterating through the globally sorted master_event_list.
        edge_list.sort(key=lambda x: x[2]) # Sort by timestamp (the 3rd element)
            
    # 7. Prepare the final graph data dictionary
    final_graph_data = {
        'apk_map': apk_map,
        'syscall_map': syscall_map,
        'binder_map': binder_map,
        'composite_map': composite_map,
        **edge_data
    }
    
    # 8. Save the data to disk in chunks
    print("Step 7: Saving graph data to chunked files...")
    save_graph_data_in_chunks(final_graph_data, output_directory)
    
    print("\n--- Unified Graph Data Preparation Complete! ---")
    return output_directory

if __name__ == '__main__':
    # --- Setup for Demonstration ---
    log_dir = "apk_pkl_logs"
    create_dummy_pkl_files(directory=log_dir, num_files=50)
    
    # --- Main Execution ---
    # This single function now handles everything and produces a directory
    # of chunked data files, ready for a GNN.
    chunk_dir = prepare_graph_data(log_dir)
    
    # --- Verification (Optional) ---
    if chunk_dir and os.path.exists(chunk_dir):
        print("\n--- Verifying contents of the chunked data directory ---")
        with open(os.path.join(chunk_dir, 'node_maps.pkl'), 'rb') as f:
            node_maps = pickle.load(f)
            print(f"Loaded node maps. Example APK map size: {len(node_maps['apk_map'])}")

        chunk_file_example = os.path.join(chunk_dir, 'app_syscall_edges_chunk_0.pkl')
        if os.path.exists(chunk_file_example):
            with open(chunk_file_example, 'rb') as f:
                edge_chunk = pickle.load(f)
                print(f"Loaded one edge chunk. Size: {len(edge_chunk)} edges.")
                print(f"Example edge triplet: {edge_chunk[0]}")

    # --- Cleanup ---
    print("\nCleaning up dummy files...")
    for f in os.listdir(log_dir):
        os.remove(os.path.join(log_dir, f))
    os.rmdir(log_dir)
    if chunk_dir and os.path.exists(chunk_dir):
        for f in os.listdir(chunk_dir):
            os.remove(os.path.join(chunk_dir, f))
        os.rmdir(chunk_dir)
    
    print("Cleanup complete.")
