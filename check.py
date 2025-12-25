import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from config.config import PARQUET_PATH

def inspect_data():
    if not os.path.exists(PARQUET_PATH):
        print(f"❌ File not found: {PARQUET_PATH}")
        return

    print(f"🔍 Reading {PARQUET_PATH}...\n")
    
    # Read Parquet file
    df = pd.read_parquet(PARQUET_PATH)
    
    # 1. Check Columns
    print("--- COLUMNS ---")
    print(df.columns.tolist())
    
    # 2. Check Row Count
    print(f"\n--- TOTAL ROWS: {len(df)} ---")
    
    # 3. Check Data Types & Nulls
    print("\n--- INFO ---")
    print(df.info())
    
    # 4. Preview Data
    print("\n--- FIRST 5 ROWS ---")
    print(df.head())

if __name__ == "__main__":
    inspect_data()