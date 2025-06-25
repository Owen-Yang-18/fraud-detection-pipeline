import os
import pickle
import time
import math
import heapq
from multiprocessing import Pool, cpu_count

# ===================================================================
#  Memory-Efficient Dynamic Graph Data Preparation
# ===================================================================

def classify_action_name(name):
    """Classifies an action name as syscall, binder, or composite."""
    if name.islower():
        return 'syscall'
    if not name.islower() and not name.isupper():
        return 'binder'
    return 'composite_behavior'

def _pass1_build_node_maps(data_directory):
    """
    Pass 1: Scans all files to build global node-to-ID mappings.
    This is memory-efficient as it only stores unique names.
    """
    print("\n--- Pass 1: Building Node Mappings (Memory-Efficient) ---")
    pkl_files = [f for f in os.listdir(data_directory) if f.endswith('.pkl')]
    if not pkl_files:
        raise FileNotFoundError(f"No .pkl files found in '{data_directory}'")
        
    all_apks = sorted([os.path.splitext(f)[0] for f in pkl_files])
    all_actions = set()

    print(f"Scanning {len(pkl_files)} files to find all unique nodes...")
    for i, pkl_file in enumerate(pkl_files):
        if (i + 1) % 1000 == 0:
            print(f"  ...scanned {i+1}/{len(pkl_files)} files for metadata...")
        filepath = os.path.join(data_directory, pkl_file)
        try:
            with open(filepath, 'rb') as f:
                events = pickle.load(f)
                for _, action_name in events:
                    all_actions.add(action_name)
        except Exception as e:
            print(f"Warning: Could not process file {pkl_file} for mapping. Error: {e}")

    apk_map = {name: i for i, name in enumerate(all_apks)}
    syscall_nodes = {name for name in all_actions if classify_action_name(name) == 'syscall'}
    binder_nodes = {name for name in all_actions if classify_action_name(name) == 'binder'}
    composite_nodes = {name for name in all_actions if classify_action_name(name) == 'composite_behavior'}

    node_maps = {
        'apk_map': apk_map,
        'syscall_map': {name: i for i, name in enumerate(sorted(list(syscall_nodes)))},
        'binder_map': {name: i for i, name in enumerate(sorted(list(binder_nodes)))},
        'composite_map': {name: i for i, name in enumerate(sorted(list(composite_nodes)))}
    }
    
    return node_maps

def _pass2_process_single_file(args):
    """
    Pass 2 worker: Processes one pkl file. It converts event names to integer IDs
    and sorts the events for that single file. This creates a small, sorted
    temporary file that is ready for merging.
    """
    filepath, node_maps, temp_dir = args
    apk_id = os.path.splitext(os.path.basename(filepath))[0]
    
    if apk_id not in node_maps['apk_map']: return None
    apk_idx = node_maps['apk_map'][apk_id]
    
    try:
        with open(filepath, 'rb') as f:
            events = pickle.load(f)
    except Exception: return None

    processed_events = []
    for timestamp, action_name in events:
        action_type = classify_action_name(action_name)
        if action_type == 'syscall' and action_name in node_maps['syscall_map']:
            action_idx = node_maps['syscall_map'][action_name]
        elif action_type == 'binder' and action_name in node_maps['binder_map']:
            action_idx = node_maps['binder_map'][action_name]
        elif action_type == 'composite_behavior' and action_name in node_maps['composite_map']:
            action_idx = node_maps['composite_map'][action_name]
        else: continue
        
        # Event format: (timestamp, source_id, dest_id, dest_type)
        processed_events.append((timestamp, apk_idx, action_idx, action_type))

    processed_events.sort(key=lambda x: x[0])

    temp_filepath = os.path.join(temp_dir, f"{apk_id}.temp.pkl")
    with open(temp_filepath, 'wb') as f:
        pickle.dump(processed_events, f)
    
    return temp_filepath

def _pass3_merge_and_save_chunks(temp_files, node_maps, output_directory, chunk_size=1000000):
    """
    Pass 3: Implements an external merge sort using a min-heap. This is the
    most memory-efficient way to sort a dataset that is too large to fit in memory.
    It reads the pre-sorted temp files and builds the final, globally-sorted chunks.
    """
    print("\n--- Pass 3: Merging All Files and Saving Final Chunks ---")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    with open(os.path.join(output_directory, 'node_maps.pkl'), 'wb') as f:
        pickle.dump(node_maps, f)
    print(f"Saved global node mappings to '{os.path.join(output_directory, 'node_maps.pkl')}'")

    min_heap = []
    file_handlers = [open(f, 'rb') for f in temp_files]
    event_iterators = [pickle.load(fh) for fh in file_handlers]

    print("Initializing k-way merge with a min-heap to ensure global time order...")
    for i, iterator in enumerate(event_iterators):
        try:
            first_event = iterator[0]
            # Heap item: (timestamp, file_index, event_index_in_file)
            heapq.heappush(min_heap, (first_event[0], i, 0))
        except IndexError: continue

    edge_data = {'app_syscall': [], 'app_binder': [], 'app_composite': []}
    chunk_counters = {k: 0 for k in edge_data}
    total_events_in_memory = 0
    total_events_processed = 0

    while min_heap:
        ts, file_idx, event_idx = heapq.heappop(min_heap)
        total_events_processed += 1
        if total_events_processed % 500000 == 0:
            print(f"  ...globally processed {total_events_processed} events...")

        timestamp, src_id, dst_id, dst_type = event_iterators[file_idx][event_idx]
        
        if dst_type == 'syscall':
            edge_data['app_syscall'].append((src_id, dst_id, timestamp))
        elif dst_type == 'binder':
            edge_data['app_binder'].append((src_id, dst_id, timestamp))
        elif dst_type == 'composite_behavior':
            edge_data['app_composite'].append((src_id, dst_id, timestamp))
        total_events_in_memory += 1
        
        next_event_idx = event_idx + 1
        if next_event_idx < len(event_iterators[file_idx]):
            next_event = event_iterators[file_idx][next_event_idx]
            heapq.heappush(min_heap, (next_event[0], file_idx, next_event_idx))
            
        if total_events_in_memory >= chunk_size:
            print(f"  - Memory threshold reached. Flushing {total_events_in_memory} events to chunks...")
            for edge_type, edge_list in edge_data.items():
                if edge_list:
                    chunk_filename = f"{edge_type}_edges_chunk_{chunk_counters[edge_type]}.pkl"
                    with open(os.path.join(output_directory, chunk_filename), 'wb') as f:
                        pickle.dump(edge_list, f)
                    edge_data[edge_type] = []
                    chunk_counters[edge_type] += 1
            total_events_in_memory = 0
    
    print("Flushing final remaining events...")
    for edge_type, edge_list in edge_data.items():
        if edge_list:
            chunk_filename = f"{edge_type}_edges_chunk_{chunk_counters[edge_type]}.pkl"
            with open(os.path.join(output_directory, chunk_filename), 'wb') as f:
                pickle.dump(edge_list, f)

    for fh in file_handlers: fh.close()


def prepare_graph_data(data_directory, output_directory="graph_data_chunks"):
    """
    Orchestrates the multi-pass, memory-efficient graph data preparation.
    """
    temp_dir = os.path.join(data_directory, "temp_processing")
    if not os.path.exists(temp_dir): os.makedirs(temp_dir)
        
    try:
        # Pass 1: Build global node maps. This is fast and low-memory.
        node_maps = _pass1_build_node_maps(data_directory)
        
        # Pass 2: In parallel, create a small, sorted temp file for each input .pkl file.
        # This reads each input file only ONCE.
        print("\n--- Pass 2: Creating Per-File Sorted Chunks (Parallel) ---")
        pkl_files = [os.path.join(data_directory, f) for f in os.listdir(data_directory) if f.endswith('.pkl')]
        
        pool_args = [(f, node_maps, temp_dir) for f in pkl_files]
        with Pool(cpu_count()) as pool:
            temp_files_created = pool.map(_pass2_process_single_file, pool_args)
        
        temp_files_created = [f for f in temp_files_created if f is not None]
        print(f"Created {len(temp_files_created)} temporary sorted files.")

        # Pass 3: Use a k-way merge (via a min-heap) to efficiently combine the sorted
        # temp files into final, globally-sorted chunks. This avoids loading all data.
        if temp_files_created:
            _pass3_merge_and_save_chunks(temp_files_created, node_maps, output_directory)
        else:
            print("No temporary files were created, skipping merge step.")

    finally:
        # Cleanup temporary files
        print("\nCleaning up temporary directory...")
        if os.path.exists(temp_dir):
            for f in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, f))
            os.rmdir(temp_dir)
    
    print("\n--- Unified Graph Data Preparation Complete! ---")
    return output_directory

if __name__ == '__main__':
    # You would replace 'apk_pkl_logs' with the path to your directory of 11,000 files
    log_dir = "apk_pkl_logs"
    output_dir = "graph_data_chunks"
    
    # prepare_graph_data(log_dir, output_directory=output_dir, num_chunks=100) # Example for real run
    
    print("This script is designed to run on a directory of .pkl files.")
    print("A production run has been commented out. No dummy data will be created.")

