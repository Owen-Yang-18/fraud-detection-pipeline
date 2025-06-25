import json
import os
import pickle
import random
from multiprocessing import Pool, cpu_count

def create_dummy_maldroid_json_files(directory="apk_json_logs", num_files=50):
    """
    Generates a directory of dummy JSON files that accurately mimics the 
    nested structure of the CICMalDroid 2020 dataset.
    """
    print(f"Creating {num_files} dummy JSON files in '{directory}'...")
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    for i in range(num_files):
        filename = os.path.join(directory, f"apk_{i}.json")
        host_events = []
        
        # Simulate a SYSCALL event
        host_events.append({
            "class": "SYSCALL",
            "low": [{"ts": 161582.4311 + i, "sysname": "write"}]
        })
        
        # Simulate a BINDER event
        host_events.append({
            "class": "BINDER",
            "method": "sendSMS",
            "low": [{"ts": 161583.0157 + i}]
        })

        # Simulate a Composite Behavior event
        host_events.append({
            "class": "FS_ACCESS(WRITE)",
            "low": [{"ts": 161591.5543 + i}]
        })
        
        # Simulate another SYSCALL to test ordering
        host_events.append({
            "class": "SYSCALL",
            "low": [{"ts": 161582.9999 + i, "sysname": "openat"}]
        })

        full_json_structure = {
            "behaviors": {"dynamic": {"host": host_events}}
        }
        
        with open(filename, 'w') as f:
            json.dump(full_json_structure, f, indent=2)
            
    print("Dummy JSON files created successfully.")


def parse_maldroid_json_to_events(json_filepath):
    """
    Parses a single CICMalDroid 2020 JSON file into a time-ordered event list.

    Args:
        json_filepath (str): The path to the input JSON file.

    Returns:
        str: The path to the saved .pkl file, or None if parsing fails.
    """
    # This function is now designed to be called by worker processes.
    # Print statements are kept minimal to avoid cluttering the console output.
    
    try:
        with open(json_filepath, 'r') as f:
            data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        # Silently fail for a single bad file in parallel mode
        return None

    event_list = []
    try:
        host_events = data['behaviors']['dynamic']['host']
    except KeyError:
        return None

    for action in host_events:
        try:
            timestamp = action['low'][0]['ts']
            action_class = action.get('class')
            action_name = None

            if action_class == 'BINDER':
                action_name = action.get('method')
            elif action_class == 'SYSCALL':
                action_name = action['low'][0].get('sysname')
            elif action_class:
                action_name = action_class
            
            if action_name:
                event_list.append((timestamp, action_name))
        except (KeyError, IndexError, TypeError):
            continue
            
    event_list.sort(key=lambda x: x[0])
    
    if event_list:
        base_filename = os.path.splitext(json_filepath)[0]
        output_filepath = f"{base_filename}.pkl"
        
        try:
            with open(output_filepath, 'wb') as f:
                pickle.dump(event_list, f)
            return output_filepath
        except Exception:
            return None
    
    return None

def parallel_process_directory(directory):
    """
    Finds all .json files in a directory and processes them in parallel.
    """
    print(f"\nScanning for JSON files in '{directory}'...")
    json_files = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.json')]
    
    if not json_files:
        print("No JSON files found to process.")
        return

    # Use all available CPU cores, or you can specify a number
    num_processes = cpu_count()
    print(f"Found {len(json_files)} files. Starting parallel processing with {num_processes} cores...")

    # Create a pool of worker processes
    with Pool(processes=num_processes) as pool:
        # map() applies the function to each item in the list and blocks until all are done
        results = pool.map(parse_maldroid_json_to_events, json_files)
    
    # Filter out any 'None' results from failed files
    successful_files = [res for res in results if res is not None]
    
    print("\n--- Parallel Processing Complete ---")
    print(f"Successfully processed {len(successful_files)} out of {len(json_files)} files.")
    if successful_files:
        print(f"Processed .pkl files have been saved in the same directory: '{directory}'")

if __name__ == '__main__':
    # --- Setup for Demonstration ---
    # 1. Create a directory with multiple dummy JSON files to simulate your environment.
    log_dir = "apk_json_logs"
    create_dummy_maldroid_json_files(directory=log_dir, num_files=50)
    
    # --- Main Execution ---
    # 2. Run the parallel processing function on the directory.
    #    This is the function you would point to your directory of 11,000 files.
    parallel_process_directory(log_dir)
    
    # --- Cleanup ---
    # 3. Clean up the created dummy files.
    print("\nCleaning up dummy files...")
    for f in os.listdir(log_dir):
        os.remove(os.path.join(log_dir, f))
    os.rmdir(log_dir)
    
    print("Cleanup complete.")
