import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from pathlib import Path
from tqdm import tqdm
import concurrent.futures
import logging

# --- Configuration ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Performs advanced feature engineering on the dataframe.
    - Creates time-based cyclical features for positional embedding.
    - Cleans IP addresses to a standard IPv4 format.
    
    Args:
        df (pd.DataFrame): The filtered dataframe.

    Returns:
        pd.DataFrame: The dataframe with new, engineered features.
    """
    df = df.copy()

    if 'timestamp' in df.columns:
        # This handles the 'MM/DD/YYYY HH:MM:SS.XXXXX' format.
        # The `format` string is explicit, but pandas is often smart enough to infer it.
        # `errors='coerce'` will turn any unparseable dates into NaT (Not a Time).
        df['timestamp_dt'] = pd.to_datetime(df['timestamp'], format='%m/%d/%Y %H:%M:%S.%f', errors='coerce')

        # Drop rows where the timestamp could not be parsed, as they are unusable for time features.
        df.dropna(subset=['timestamp_dt'], inplace=True)

        # Extract components for cyclical feature creation
        dt_series = df['timestamp_dt']
        seconds = dt_series.dt.second + dt_series.dt.microsecond / 1_000_000
        minutes = dt_series.dt.minute
        hours = dt_series.dt.hour
        days_of_week = dt_series.dt.dayofweek # Monday=0, Sunday=6
        months = dt_series.dt.month

        # Create cyclical "positional embedding" features using sine and cosine
        # This helps the model understand cyclical patterns.
        df['second_sin'] = np.sin(2 * np.pi * seconds / 60)
        df['second_cos'] = np.cos(2 * np.pi * seconds / 60)
        
        df['minute_sin'] = np.sin(2 * np.pi * minutes / 60)
        df['minute_cos'] = np.cos(2 * np.pi * minutes / 60)
        
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24)
        
        df['day_of_week_sin'] = np.sin(2 * np.pi * days_of_week / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * days_of_week / 7)
        
        df['month_sin'] = np.sin(2 * np.pi * months / 12)
        df['month_cos'] = np.cos(2 * np.pi * months / 12)

        # Clean up the intermediate datetime column
        df.drop(columns=['timestamp_dt'], inplace=True)

    # IP address cleaning logic remains the same
    ip_cols = ['local_address', 'remote_address']
    for col in ip_cols:
        if col in df.columns:
            df[col] = df[col].fillna('').astype(str)
            df[col] = df[col].str.split(':').str[-1]
            df[col].replace('', pd.NA, inplace=True)
            
    return df

def process_single_parquet(file_path: Path, columns_to_keep: list) -> pd.DataFrame | None:
    try:
        df = pd.read_parquet(file_path)
        available_columns = [col for col in columns_to_keep if col in df.columns]
        if not available_columns:
            logging.warning(f"Skipping {file_path}: None of the desired columns found.")
            return None
        
        # Ensure the timestamp column exists before trying to keep it
        if 'timestamp' not in df.columns:
            logging.warning(f"Skipping {file_path}: 'timestamp' column not found.")
            return None

        df_filtered = df[available_columns].copy()
        
        # Apply the feature engineering and cleaning function
        df_engineered = engineer_features(df_filtered)
        
        return df_engineered
    except Exception as e:
        logging.error(f"Error processing file {file_path}: {e}")
        return None

def process_log_directory(root_dir: str, columns_to_keep: list, output_path: str) -> None:
    root_path = Path(root_dir)
    if not root_path.is_dir():
        logging.error(f"Root directory not found: {root_dir}")
        return

    logging.info(f"Starting to scan for Parquet files in: {root_dir}")
    parquet_files = list(root_path.rglob('*.parquet'))
    logging.info(f"Found {len(parquet_files)} Parquet files to process.")

    if not parquet_files:
        logging.warning("No Parquet files found. Exiting.")
        return

    processed_dfs = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        future_to_file = {executor.submit(process_single_parquet, file, columns_to_keep): file for file in parquet_files}
        for future in tqdm(concurrent.futures.as_completed(future_to_file), total=len(parquet_files), desc="Processing Parquet Files"):
            result_df = future.result()
            if result_df is not None and not result_df.empty:
                processed_dfs.append(result_df)

    if not processed_dfs:
        logging.warning("No data was processed successfully. No output file will be created.")
        return

    final_df = pd.concat(processed_dfs, ignore_index=True)
    if 'timestamp' in final_df.columns:
        # Sort by the original string timestamp to maintain order before saving
        final_df = final_df.sort_values(by='timestamp').reset_index(drop=True)

    final_df.to_parquet(output_path, index=False, compression='snappy')
    logging.info(f"Consolidated data saved to {output_path}")

# --- Helper Function to Create a Dummy Dataset for Testing ---
def create_dummy_dataset(base_dir="dummy_data", num_files=5):
    """Creates a mock directory structure and Parquet files for demonstration."""
    logging.info(f"Creating dummy dataset in '{base_dir}'...")
    base_path = Path(base_dir)
    
    # Generate timestamps in the specified string format
    timestamps = pd.to_datetime(pd.date_range('2024-01-21 02:05:00', periods=10, freq='s'))
    data = {
        'event_id': range(10),
        'timestamp': [t.strftime('%m/%d/%Y %H:%M:%S.%f')[:-3] for t in timestamps], # Format to MM/DD/YYYY...
        'name': ['PROCESS_CREATION', 'SOCKET_CREATION'] * 5,
        'pid': [100, 101, 102, 103, 104] * 2,
    }
    df = pd.DataFrame(data)

    for i in range(num_files):
        dir_path = base_path / f"part-{i}"
        dir_path.mkdir(parents=True, exist_ok=True)
        file_path = dir_path / "data.parquet"
        df.to_parquet(file_path)
    logging.info(f"Dummy dataset created with {num_files} files.")


# --- Main Execution Block ---
if __name__ == '__main__':
    DATA_ROOT = "dummy_data"
    OUTPUT_FILE = "processed_event_logs_with_time_features.parquet"
    COLUMNS_TO_KEEP = ['timestamp', 'name', 'pid', 'local_address', 'remote_address']
    
    # Create a dummy dataset to run the script on
    create_dummy_dataset(DATA_ROOT)

    # Run the main processing pipeline
    process_log_directory(root_dir=DATA_ROOT, columns_to_keep=COLUMNS_TO_KEEP, output_path=OUTPUT_FILE)
    
    # Verification Step
    logging.info(f"\n--- Verification ---")
    if Path(OUTPUT_FILE).exists():
        final_df = pd.read_parquet(OUTPUT_FILE)
        logging.info(f"Successfully loaded output file: {OUTPUT_FILE}")
        logging.info("Head of the final dataframe:")
        print(final_df.head())
        logging.info("\nColumns in the final dataframe:")
        print(final_df.columns.tolist())
    else:
        logging.error("Output file was not created.")

