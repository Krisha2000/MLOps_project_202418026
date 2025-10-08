import pandas as pd
from pathlib import Path
import time

def load_data_to_feature_store():
    """
    Reads the raw CSV data, prepares it for Feast, and saves it as a Parquet file.
    """
    try:
        # Define paths
        project_root = Path(__file__).parent.parent
        data_dir = project_root / "data"
        feature_repo_dir = project_root / "feature_repo"
        
        # Load the raw CSV data
        print("Loading raw data from train.csv...")
        df = pd.read_csv(data_dir / "train.csv")
        
        # --- Feature Engineering & Preparation ---
        # 1. Create a dummy event timestamp. In a real scenario, this would be the
        #    timestamp of when the application was submitted.
        #    We'll create a staggered timestamp for each row.
        now = pd.to_datetime('now', utc=True)
        df['event_timestamp'] = [now - pd.Timedelta(seconds=i) for i in range(len(df), 0, -1)]
        
        # 2. Create a created timestamp.
        df['created_timestamp'] = now
        
        # Define the path for the output Parquet file
        output_path = feature_repo_dir / "data" / "train.parquet"
        print(f"Preparing to save data to {output_path}...")
        
        # Ensure the target directory exists
        output_path.parent.mkdir(exist_ok=True)
        
        # Save the DataFrame to a Parquet file
        df.to_parquet(output_path)
        
        print(f"Successfully converted and saved data to {output_path}")
        print("This file will be used as the source for our Feast feature store.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    load_data_to_feature_store()