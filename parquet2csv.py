import pandas as pd

def parquet_to_csv(parquet_path, csv_path):
    """
    Convert a .parquet file to a .csv file.
    
    Args:
        parquet_path (str): Path to the input .parquet file.
        csv_path (str): Path to save the output .csv file.
    """
    df = pd.read_parquet(parquet_path)
    df.to_csv(csv_path, index=False)
    print(f"Successfully converted {parquet_path} to {csv_path}")

# Example usage
if __name__ == "__main__":
    parquet_path = "data/final/raw_dataset.parquet"
    csv_path = "data/final/raw_dataset.csv"
    parquet_to_csv(parquet_path, csv_path)
