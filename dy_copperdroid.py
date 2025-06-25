import pickle
import heapq
import os
from pathlib import Path
from typing import Iterator, Tuple, Any
import gc


class PickleFileIterator:
    """Iterator wrapper for pickle files to handle exhausted files gracefully."""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.file_handle = None
        self.data = None
        self.index = 0
        self.exhausted = False
        
    def __enter__(self):
        self.file_handle = open(self.filepath, 'rb')
        self.data = pickle.load(self.file_handle)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.file_handle:
            self.file_handle.close()
            
    def __iter__(self):
        return self
        
    def __next__(self):
        if self.exhausted or self.index >= len(self.data):
            self.exhausted = True
            raise StopIteration
        
        item = self.data[self.index]
        self.index += 1
        return item


def merge_pkl_files_heap_approach(
    input_folder: str, 
    output_file: str, 
    chunk_size: int = 100000
) -> None:
    """
    Merge sorted pkl files using heap-based k-way merge.
    
    Args:
        input_folder: Path to folder containing pkl files
        output_file: Output file path for merged results
        chunk_size: Number of records to write at once
    """
    pkl_files = list(Path(input_folder).glob("*.pkl"))
    
    if not pkl_files:
        print("No pkl files found!")
        return
    
    print(f"Found {len(pkl_files)} pkl files to merge")
    
    # Initialize heap with first element from each file
    heap = []
    iterators = []
    
    # Open all files and get first elements
    for i, pkl_file in enumerate(pkl_files):
        try:
            iterator = PickleFileIterator(str(pkl_file))
            iterator.__enter__()
            iterators.append(iterator)
            
            # Get first element
            first_item = next(iterator)
            timestamp, action_name = first_item
            # Push (timestamp, file_index, action_name) to heap
            heapq.heappush(heap, (timestamp, i, action_name))
            
        except (StopIteration, EOFError):
            # File is empty, skip it
            if iterator:
                iterator.__exit__(None, None, None)
            continue
    
    # Merge and write results
    merged_results = []
    total_processed = 0
    
    try:
        with open(output_file, 'wb') as outfile:
            while heap:
                # Get smallest timestamp
                timestamp, file_idx, action_name = heapq.heappop(heap)
                merged_results.append((timestamp, action_name))
                total_processed += 1
                
                # Write chunk to file when buffer is full
                if len(merged_results) >= chunk_size:
                    pickle.dump(merged_results, outfile)
                    merged_results = []
                    
                    if total_processed % 50000 == 0:
                        print(f"Processed {total_processed} records...")
                
                # Get next element from the same file
                try:
                    next_item = next(iterators[file_idx])
                    next_timestamp, next_action_name = next_item
                    heapq.heappush(heap, (next_timestamp, file_idx, next_action_name))
                except StopIteration:
                    # This file is exhausted, continue with others
                    pass
            
            # Write remaining results
            if merged_results:
                pickle.dump(merged_results, outfile)
                
    finally:
        # Clean up all file handles
        for iterator in iterators:
            try:
                iterator.__exit__(None, None, None)
            except:
                pass
    
    print(f"Merge complete! Processed {total_processed} total records")


def merge_pkl_files_batch_approach(
    input_folder: str, 
    output_folder: str, 
    files_per_batch: int = 100,
    records_per_chunk: int = 100000,
    save_as_chunks: bool = True
) -> None:
    """
    Alternative approach: Merge in batches to avoid file handle limits.
    
    Args:
        input_folder: Path to folder containing pkl files  
        output_folder: Output folder for intermediate and final files
        files_per_batch: Number of files to merge at once
        records_per_chunk: Records per output chunk
        save_as_chunks: If True, save final result as separate chunk files
    """
    os.makedirs(output_folder, exist_ok=True)
    pkl_files = list(Path(input_folder).glob("*.pkl"))
    
    print(f"Found {len(pkl_files)} pkl files")
    print(f"Processing in batches of {files_per_batch} files")
    
    # Phase 1: Merge files in batches
    batch_files = []
    batch_num = 0
    
    for i in range(0, len(pkl_files), files_per_batch):
        batch = pkl_files[i:i + files_per_batch]
        batch_output = os.path.join(output_folder, f"batch_{batch_num:04d}.pkl")
        
        print(f"Processing batch {batch_num + 1}: {len(batch)} files")
        
        # Merge this batch
        _merge_batch(batch, batch_output, records_per_chunk)
        batch_files.append(batch_output)
        batch_num += 1
        
        # Force garbage collection
        gc.collect()
    
    # Phase 2: Merge all batch files
    print(f"Merging {len(batch_files)} batch files into final result")
    
    if save_as_chunks:
        # Save as separate chunk files
        _merge_to_chunks(batch_files, output_folder, records_per_chunk)
    else:
        # Save as single file
        final_output = os.path.join(output_folder, "final_merged.pkl")
        _merge_batch(batch_files, final_output, records_per_chunk)
    
    # Clean up intermediate files
    for batch_file in batch_files:
        os.remove(batch_file)
    
    print("Batch merge complete!")


def _merge_to_chunks(file_list: list, output_folder: str, chunk_size: int) -> None:
    """Merge files and save as separate chunk files."""
    heap = []
    iterators = []
    
    # Open files and initialize heap
    for i, filepath in enumerate(file_list):
        try:
            iterator = PickleFileIterator(str(filepath))
            iterator.__enter__()
            iterators.append(iterator)
            
            first_item = next(iterator)
            timestamp, action_name = first_item
            heapq.heappush(heap, (timestamp, i, action_name))
            
        except (StopIteration, EOFError):
            if iterator:
                iterator.__exit__(None, None, None)
            continue
    
    # Merge and save as chunks
    merged_results = []
    chunk_number = 0
    total_processed = 0
    
    try:
        while heap:
            timestamp, file_idx, action_name = heapq.heappop(heap)
            merged_results.append((timestamp, action_name))
            total_processed += 1
            
            # Save chunk when buffer is full
            if len(merged_results) >= chunk_size:
                chunk_filename = os.path.join(output_folder, f"chunk_{chunk_number:06d}.pkl")
                with open(chunk_filename, 'wb') as chunk_file:
                    pickle.dump(merged_results, chunk_file)
                
                print(f"Saved chunk {chunk_number}: {len(merged_results)} records")
                merged_results = []
                chunk_number += 1
                
                if total_processed % 50000 == 0:
                    print(f"Processed {total_processed} records...")
            
            try:
                next_item = next(iterators[file_idx])
                next_timestamp, next_action_name = next_item
                heapq.heappush(heap, (next_timestamp, file_idx, next_action_name))
            except StopIteration:
                pass
        
        # Save final chunk if there are remaining results
        if merged_results:
            chunk_filename = os.path.join(output_folder, f"chunk_{chunk_number:06d}.pkl")
            with open(chunk_filename, 'wb') as chunk_file:
                pickle.dump(merged_results, chunk_file)
            print(f"Saved final chunk {chunk_number}: {len(merged_results)} records")
            chunk_number += 1
        
        print(f"Created {chunk_number} chunk files with {total_processed} total records")
                
    finally:
        for iterator in iterators:
            try:
                iterator.__exit__(None, None, None)
            except:
                pass


def _merge_batch(file_list: list, output_file: str, chunk_size: int) -> None:
    """Helper function to merge a batch of files."""
    heap = []
    iterators = []
    
    # Open files and initialize heap
    for i, filepath in enumerate(file_list):
        try:
            iterator = PickleFileIterator(str(filepath))
            iterator.__enter__()
            iterators.append(iterator)
            
            first_item = next(iterator)
            timestamp, action_name = first_item
            heapq.heappush(heap, (timestamp, i, action_name))
            
        except (StopIteration, EOFError):
            if iterator:
                iterator.__exit__(None, None, None)
            continue
    
    # Merge
    merged_results = []
    
    try:
        with open(output_file, 'wb') as outfile:
            while heap:
                timestamp, file_idx, action_name = heapq.heappop(heap)
                merged_results.append((timestamp, action_name))
                
                if len(merged_results) >= chunk_size:
                    pickle.dump(merged_results, outfile)
                    merged_results = []
                
                try:
                    next_item = next(iterators[file_idx])
                    next_timestamp, next_action_name = next_item
                    heapq.heappush(heap, (next_timestamp, file_idx, next_action_name))
                except StopIteration:
                    pass
            
            if merged_results:
                pickle.dump(merged_results, outfile)
                
    finally:
        for iterator in iterators:
            try:
                iterator.__exit__(None, None, None)
            except:
                pass


# Example usage
if __name__ == "__main__":
    # Approach 1: Direct heap merge (if system can handle 12000 file handles)
    # merge_pkl_files_heap_approach(
    #     input_folder="path/to/pkl/files",
    #     output_file="merged_output.pkl"
    # )
    
    # Approach 2a: Batch processing with separate chunk files (recommended)
    merge_pkl_files_batch_approach(
        input_folder="path/to/pkl/files",
        output_folder="output",
        files_per_batch=50,
        records_per_chunk=100000,
        save_as_chunks=True  # Creates chunk_000000.pkl, chunk_000001.pkl, etc.
    )
    
    # Approach 2b: Batch processing with single output file
    # merge_pkl_files_batch_approach(
    #     input_folder="path/to/pkl/files",
    #     output_folder="output", 
    #     files_per_batch=50,
    #     records_per_chunk=100000,
    #     save_as_chunks=False  # Creates single final_merged.pkl
    # )


# Utility function to read the merged results
def read_merged_results(filepath: str) -> Iterator[Tuple[Any, str]]:
    """
    Generator to read merged results from single file without loading everything into memory.
    """
    with open(filepath, 'rb') as f:
        try:
            while True:
                chunk = pickle.load(f)
                for item in chunk:
                    yield item
        except EOFError:
            pass


def read_chunk_results(chunk_folder: str) -> Iterator[Tuple[Any, str]]:
    """
    Generator to read merged results from chunk files in order.
    """
    chunk_files = sorted(Path(chunk_folder).glob("chunk_*.pkl"))
    
    for chunk_file in chunk_files:
        with open(chunk_file, 'rb') as f:
            chunk_data = pickle.load(f)
            for item in chunk_data:
                yield item


# Example of reading results
def print_sample_results(filepath_or_folder: str, num_samples: int = 10, is_chunks: bool = True):
    """Print first few results to verify merge worked correctly."""
    print(f"First {num_samples} merged records:")
    
    if is_chunks:
        iterator = read_chunk_results(filepath_or_folder)
    else:
        iterator = read_merged_results(filepath_or_folder)
    
    for i, (timestamp, action_name) in enumerate(iterator):
        if i >= num_samples:
            break
        print(f"  {timestamp}: {action_name}")