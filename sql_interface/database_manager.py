"""
SQLite Database Manager

STUDENT TASK (15 points):
Implement local database storage and retrieval
"""

import sqlite3
import pandas as pd
import os
import sys

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH, PARQUET_PATH

class DatabaseManager:
    def __init__(self, db_path=None):
        """Initialize Database Connection"""
        self.db_path = db_path or DB_PATH
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        print(f"🔌 Database initialized at: {self.db_path}")

    def get_connection(self):
        """Create a database connection"""
        return sqlite3.connect(self.db_path)

    def create_tables(self):
        """
        Create the stock_data table schema
        """
        print("   -> Creating tables...")
        # Schema definition from Task 3 requirements
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS stock_data (
            ticker TEXT NOT NULL,
            date DATETIME NOT NULL,
            open REAL,
            high REAL,
            low REAL,
            close REAL,
            volume INTEGER,
            ma_7 REAL,
            ma_30 REAL,
            ma_90 REAL,
            rsi REAL,
            volatility REAL,
            daily_return REAL,
            sharpe_ratio REAL,
            PRIMARY KEY (ticker, date)
        );
        """
        
        # Create index for faster queries
        create_index_sql = "CREATE INDEX IF NOT EXISTS idx_ticker_date ON stock_data (ticker, date);"
        
        try:
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(create_table_sql)
                cursor.execute(create_index_sql)
                conn.commit()
            print("   -> Tables created successfully.")
        except Exception as e:
            print(f"❌ Error creating tables: {e}")

    def load_from_parquet(self, parquet_path=None):
        """
        Load processed Parquet data into SQLite
        """
        path = parquet_path or PARQUET_PATH
        print(f"📥 Loading data from {path}...")
        
        if not os.path.exists(path):
            print(f"❌ Error: Parquet file not found at {path}")
            return

        try:
            # Read Parquet using Pandas
            df = pd.read_parquet(path)
            
            # Normalize columns to lowercase to match SQL schema
            df.columns = [c.lower() for c in df.columns]
            
            # Ensure Date is string format for SQLite
            df['date'] = df['date'].astype(str)
            
            # Write to SQLite
            with self.get_connection() as conn:
                df.to_sql('stock_data', conn, if_exists='replace', index=False)
                
            print(f"✅ Loaded {len(df)} rows into database.")
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")

    def get_stock_data(self, ticker, start_date=None, end_date=None):
        """
        Query stock data for a specific ticker
        """
        query = "SELECT * FROM stock_data WHERE ticker = ?"
        params = [ticker]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date)
        if end_date:
            query += " AND date <= ?"
            params.append(end_date)
            
        query += " ORDER BY date ASC"
        
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn, params=params)

    def get_latest_prices(self):
        """
        Get the most recent close price for all tickers
        """
        # Subquery to find the latest date in the database
        query = """
        SELECT ticker, date, close, rsi, daily_return 
        FROM stock_data 
        WHERE date = (SELECT MAX(date) FROM stock_data)
        ORDER BY ticker
        """
        with self.get_connection() as conn:
            return pd.read_sql_query(query, conn)


def run_database_setup():
    """Main function to setup database"""
    print("="*60)
    print("🗄️  DATABASE MANAGER SETUP")
    print("="*60)
    
    db = DatabaseManager()
    
    # 1. Create Schema
    db.create_tables()
    
    # 2. Load Data from Parquet (Task 2 output)
    db.load_from_parquet()
    
    # 3. Test Query
    print("\n🔍 Testing query (Latest Prices):")
    try:
        latest = db.get_latest_prices()
        print(latest)
    except Exception as e:
        print(f"Query failed: {e}")
        
    print("="*60)

if __name__ == "__main__":
    run_database_setup()