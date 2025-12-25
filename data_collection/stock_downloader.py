"""
Stock Data Downloader using Yahoo Finance API

STUDENT TASK (20 points):
Implement functions to download historical stock data from Yahoo Finance

WHAT YOU'LL LEARN:
- API integration with yfinance
- Error handling and data validation
- File I/O operations
- Data quality checks

EXPECTED OUTPUT:
- CSV files in data/stock_data/
- Format: {TICKER}_stock_data.csv
- Columns: Ticker, Date, Open, High, Low, Close, Volume
- ~1254 rows per ticker (5 years daily data)
"""

import yfinance as yf
import pandas as pd
import os
from datetime import datetime
import sys

# Add project root to path to import config
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Note: config.py uses START_DATE, ensuring we import it correctly
from config.config import STOCK_DATA_DIR, DEFAULT_TICKERS, START_DATE


class StockDownloader:
    """Downloads historical stock data from Yahoo Finance"""

    def __init__(self, tickers=None, start_date=None, end_date=None):
        """
        Initialize downloader

        TODO: Set tickers, dates, create output directory
        """
        self.tickers = tickers or DEFAULT_TICKERS
        self.start_date = start_date or START_DATE
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        os.makedirs(STOCK_DATA_DIR, exist_ok=True)
        print(f"📂 Output directory ready: {STOCK_DATA_DIR}")

    def download_stock_data(self, ticker):
        """
        Download data for ONE ticker

        Args:
            ticker: Stock symbol (e.g., 'AAPL')

        Returns:
            DataFrame with columns: Ticker, Date, Open, High, Low, Close, Volume

        TODO (10 points):
        1. Use yf.download(ticker, start=..., end=...)
        2. Check if data is empty
        3. Reset index to make Date a column
        4. Add 'Ticker' column
        5. Return DataFrame

        HINT: yf.download returns DataFrame with Date as index
        """
        print(f"Downloading {ticker}...")

        # YOUR CODE HERE
        try:
            # 1. Use yf.download
            # auto_adjust=True gets the split-adjusted price (better for analysis)
            # progress=False hides the default progress bar so we can print our own
            df = yf.download(ticker, start=self.start_date, end=self.end_date, progress=False, auto_adjust=False)

            # 2. Check if data is empty
            if df.empty:
                print(f"❌ Failed (Empty Data)")
                return None

            # Handle yfinance 0.2+ MultiIndex columns if necessary
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            # 3. Reset index to make Date a column
            df = df.reset_index()

            # 4. Add 'Ticker' column
            df['Ticker'] = ticker

            # Ensure 'Date' is datetime format (removes timezone info for compatibility)
            df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)

            # Select and reorder columns
            required_columns = ['Ticker', 'Date', 'Open', 'High', 'Low', 'Close', 'Volume']
            
            # Check if all columns exist
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"❌ Failed (Missing columns: {missing_cols})")
                return None

            df = df[required_columns]
            
            print(f"✅ Success ({len(df)} rows)")
            return df

        except Exception as e:
            print(f"❌ Error: {e}")
            return None

    def download_all(self):
        """
        Download data for ALL tickers

        Returns:
            dict: {ticker: DataFrame}

        TODO (5 points):
        1. Loop through self.tickers
        2. Call download_stock_data() for each
        3. Store in dictionary
        4. Handle failures gracefully
        """
        stock_data = {}
        print(f"\n🚀 Starting download for: {', '.join(self.tickers)}")
        print(f"📅 Date Range: {self.start_date} to {self.end_date}\n")

        # YOUR CODE HERE
        for ticker in self.tickers:
            # 2. Call download_stock_data
            df = self.download_stock_data(ticker)
            
            # 3. Store in dictionary if successful
            if df is not None:
                stock_data[ticker] = df
        
        return stock_data

    def save_to_csv(self, stock_data):
        """
        Save all data to CSV files

        Args:
            stock_data: dict of {ticker: DataFrame}

        TODO (3 points):
        1. Loop through stock_data
        2. Create filename: f"{ticker}_stock_data.csv"
        3. Save: df.to_csv(filepath, index=False)
        """
        # YOUR CODE HERE
        print("\n💾 Saving data to CSV...")
        saved_files = []
        
        # 1. Loop through stock_data
        for ticker, df in stock_data.items():
            # 2. Create filename
            filename = f"{ticker}_stock_data.csv"
            filepath = os.path.join(STOCK_DATA_DIR, filename)
            
            # 3. Save to CSV
            df.to_csv(filepath, index=False)
            saved_files.append(filepath)
            print(f"   -> Saved {filename}")
            
        return saved_files

    def validate_data(self, stock_data):
        """
        Validate data quality

        Args:
            stock_data: dict of DataFrames

        Returns:
            bool: True if all valid

        TODO (2 points):
        Check for:
        - Minimum 200 rows
        - No missing values in critical columns
        - No duplicate dates
        - Positive prices
        - No extreme outliers (>50% daily change)
        """
        # YOUR CODE HERE
        print("\n🔍 Validating data quality...")
        all_valid = True
        
        for ticker, df in stock_data.items():
            issues = []
            
            # Check 1: Minimum rows
            if len(df) < 200:
                issues.append(f"Insufficient data ({len(df)} rows)")
            
            # Check 2: Missing values
            if df.isnull().values.any():
                missing_count = df.isnull().sum().sum()
                issues.append(f"Contains {missing_count} missing values")
            
            # Check 3: Duplicate dates
            if df['Date'].duplicated().any():
                issues.append("Duplicate dates found")
            
            # Check 4: Positive prices
            if (df[['Open', 'High', 'Low', 'Close']] <= 0).any().any():
                issues.append("Non-positive stock prices found")

            if issues:
                print(f"⚠️  {ticker}: Validation Failed - {', '.join(issues)}")
                all_valid = False
            else:
                print(f"✅ {ticker}: Validated")
                
        return all_valid


def run_data_collection():
    """
    Main function - orchestrates download process

    TODO:
    1. Create StockDownloader()
    2. Call download_all()
    3. Call validate_data()
    4. Call save_to_csv()
    5. Print summary
    """
    print("="*60)
    print("STOCK DATA COLLECTION")
    print("="*60)

    # YOUR CODE HERE
    # 1. Create StockDownloader
    downloader = StockDownloader()
    
    # 2. Call download_all()
    data = downloader.download_all()
    
    if not data:
        print("\n❌ No data downloaded. Check your internet connection or ticker symbols.")
        return

    # 3. Call validate_data()
    if downloader.validate_data(data):
        print("\n✨ All data passed validation checks.")
    else:
        print("\n⚠️ Some data failed validation. Check logs above.")

    # 4. Call save_to_csv()
    downloader.save_to_csv(data)
    
    print("\n✅ Data collection complete!")
    print("="*60)


if __name__ == "__main__":
    run_data_collection()