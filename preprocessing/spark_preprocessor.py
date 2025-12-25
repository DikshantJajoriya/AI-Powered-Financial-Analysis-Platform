"""
PySpark Data Preprocessing and Feature Engineering

STUDENT TASK (30 points):
Implement PySpark-based data preprocessing and feature engineering for stock data

LEARNING OBJECTIVES:
- Load data into Spark DataFrames
- Use Window functions for time series features
- Calculate technical indicators (MA, RSI, Volatility)
- Handle missing values
- Save processed data as Parquet

EXPECTED OUTPUT:
- Parquet file: data/processed_stocks.parquet
- Columns: Ticker, Date, Open, High, Low, Close, Volume,
           MA_7, MA_30, MA_90, RSI, Volatility, Daily_Return, Sharpe_Ratio
- ~6270 rows (1254 per ticker × 5 tickers)
"""

import os
import sys
import shutil
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    STOCK_DATA_DIR, PARQUET_PATH,
    MA_WINDOWS, RSI_PERIOD, VOLATILITY_WINDOW,
    APP_NAME, SPARK_DRIVER_MEMORY
)


class SparkPreprocessor:
    """PySpark-based data preprocessing and feature engineering"""

    def __init__(self, spark=None):
        """
        Initialize preprocessor with Spark session

        Args:
            spark: SparkSession (creates new one if None)

        TODO: Students - Complete initialization
        """
        if spark is None:
            print(f"⚡ Initializing Spark Session ({APP_NAME})...")
            # Create Spark session with memory constraints
            self.spark = SparkSession.builder \
                .appName(APP_NAME) \
                .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
                .config("spark.sql.legacy.timeParserPolicy", "LEGACY") \
                .config("spark.sql.shuffle.partitions", "5") \
                .master("local[*]") \
                .getOrCreate()
        else:
            self.spark = spark
            
        self.spark.sparkContext.setLogLevel("WARN")

    def load_csv_files(self):
        """
        Load all CSV files from data/stock_data/ into a single DataFrame
        """
        print("\n📂 Loading CSV files...")
        try:
            # Read all CSVs in the directory
            df = self.spark.read.option("header", "true") \
                .option("inferSchema", "true") \
                .csv(os.path.join(STOCK_DATA_DIR, "*.csv"))
            
            # Cache data since it fits in memory (Optimization for 8GB RAM)
            df.cache()
            print(f"   -> Loaded {df.count()} rows.")
            return df
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            sys.exit(1)

    def calculate_moving_averages(self, df, windows=[7, 30, 90]):
        """
        Calculate moving averages using PySpark Window functions

        Args:
            df: Spark DataFrame
            windows: List of window sizes (e.g., [7, 30, 90])

        Returns:
            DataFrame with MA columns added

        TODO: Students - Implement this function (8 points)

        HINTS:
        1. Window specification: Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-N+1, 0)
        2. For 7-day MA: rowsBetween(-6, 0) means current row + previous 6 rows = 7 total
        3. Use F.avg('Close').over(window) to calculate average
        4. Create columns: MA_7, MA_30, MA_90
        5. Loop through window sizes to avoid code duplication

        FORMULA:
            MA_N = Average of last N closing prices
        """
        print(f"\nCalculating moving averages: {windows}")

        # YOUR CODE HERE
        # Window: Partition by Ticker, Order by Date
        # rowsBetween(-N+1, 0) looks at current row + previous N-1 rows
        for w in windows:
            window_spec = Window.partitionBy("Ticker").orderBy("Date") \
                .rowsBetween(-w + 1, 0)
            
            col_name = f"MA_{w}"
            df = df.withColumn(col_name, F.avg("Close").over(window_spec))
            print(f"   -> Created {col_name}")
            
        return df

    def calculate_rsi(self, df, window=14):
        """
        Calculate RSI (Relative Strength Index)

        Args:
            df: Spark DataFrame
            window: RSI window size (default: 14)

        Returns:
            DataFrame with RSI column added

        TODO: Students - Implement this function (10 points)

        RSI FORMULA (5 steps):
        1. Calculate price changes: Change = Close - Previous Close
        2. Separate gains and losses:
           - Gain = Change if Change > 0 else 0
           - Loss = -Change if Change < 0 else 0
        3. Calculate average gain and loss over window (14 days)
           - Avg Gain = Average of Gains over window
           - Avg Loss = Average of Losses over window
        4. Calculate RS = Avg Gain / Avg Loss
        5. Calculate RSI = 100 - (100 / (1 + RS))

        HINTS:
        1. Use F.lag('Close', 1) to get previous day's close
        2. Use F.when() for conditional logic (separate gains/losses)
        3. Use Window.partitionBy('Ticker').orderBy('Date') for each stock
        4. Use rolling window: rowsBetween(-(window-1), 0) for averages
        5. Drop intermediate columns at the end to keep DataFrame clean

        RSI INTERPRETATION:
        - RSI > 70: Overbought (potential sell signal)
        - RSI < 30: Oversold (potential buy signal)
        - RSI = 50: Neutral
        """
        print(f"\nCalculating RSI (window={window})...")

        # YOUR CODE HERE
        # 1. Calculate price changes
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        df = df.withColumn("Diff", F.col("Close") - F.lag("Close", 1).over(window_spec))
        
        # 2. Separate gains and losses
        df = df.withColumn("Gain", F.when(F.col("Diff") > 0, F.col("Diff")).otherwise(0))
        df = df.withColumn("Loss", F.when(F.col("Diff") < 0, F.abs(F.col("Diff"))).otherwise(0))
        
        # 3. Calculate Average Gain and Loss
        rsi_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-window + 1, 0)
        df = df.withColumn("Avg_Gain", F.avg("Gain").over(rsi_window))
        df = df.withColumn("Avg_Loss", F.avg("Loss").over(rsi_window))
        
        # 4. Calculate RS
        df = df.withColumn("RS", F.col("Avg_Gain") / (F.col("Avg_Loss") + 1e-9))
        
        # 5. Calculate RSI
        df = df.withColumn("RSI", 100 - (100 / (1 + F.col("RS"))))
        
        # Cleanup
        df = df.drop("Diff", "Gain", "Loss", "Avg_Gain", "Avg_Loss", "RS")
        print("   -> Created RSI column")
        
        return df

    def calculate_volatility(self, df, window=30):
        """
        Calculate rolling volatility (standard deviation of close prices)

        Args:
            df: Spark DataFrame
            window: Volatility window size (default: 30)

        Returns:
            DataFrame with Volatility column added

        TODO: Students - Implement this function (3 points)

        FORMULA:
            Volatility = Standard Deviation of Close price over rolling window

        HINTS:
        1. Use Window.partitionBy('Ticker').orderBy('Date').rowsBetween(-(window-1), 0)
        2. Use F.stddev('Close').over(window) to calculate rolling std dev
        3. Higher volatility = more risky stock
        """
        print(f"\nCalculating volatility (window={window})...")

        # YOUR CODE HERE
        window_spec = Window.partitionBy("Ticker").orderBy("Date") \
            .rowsBetween(-window + 1, 0)
        
        df = df.withColumn("Volatility", F.stddev("Close").over(window_spec))
        print("   -> Created Volatility column")
        return df

    def calculate_returns(self, df):
        """
        Calculate daily returns and Sharpe ratio

        Args:
            df: Spark DataFrame

        Returns:
            DataFrame with Daily_Return and Sharpe_Ratio columns

        TODO: Students - Implement this function (4 points)

        DAILY RETURN FORMULA:
            Daily_Return = (Close - Previous Close) / Previous Close

        SHARPE RATIO FORMULA:
            Sharpe_Ratio = Mean(Daily_Return) / Std_Dev(Daily_Return)

        HINTS:
        1. Use lag() to get previous close
        2. Use Window functions for mean and stddev
        """
        print(f"\nCalculating returns and Sharpe ratio...")


        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        # Daily Return
        df = df.withColumn("Prev_Close", F.lag("Close", 1).over(window_spec))
        df = df.withColumn("Daily_Return", (F.col("Close") - F.col("Prev_Close")) / F.col("Prev_Close"))
        
        # Sharpe Ratio (Mean / StdDev of returns over 30 days)
        sharpe_window = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(-30, 0)
        df = df.withColumn("Mean_Return", F.avg("Daily_Return").over(sharpe_window))
        df = df.withColumn("Std_Return", F.stddev("Daily_Return").over(sharpe_window))
        
        df = df.withColumn("Sharpe_Ratio", 
                           F.when(F.col("Std_Return") == 0, 0)
                           .otherwise(F.col("Mean_Return") / F.col("Std_Return")))
        
        df = df.drop("Prev_Close", "Mean_Return", "Std_Return")
        print(f"   -> Created Daily_Return and Sharpe_Ratio columns")
        return df

    def handle_missing_values(self, df):
        """
        Handle missing values in the dataset

        Args:
            df: Spark DataFrame

        Returns:
            DataFrame with missing values handled

        TODO: Students - Implement missing value handling

        STRATEGIES:
        1. Drop rows with missing critical values (Open, High, Low, Close, Volume)
        2. Forward fill missing indicator values (MA, RSI, etc.)
        3. Or drop rows with any missing values

        HINTS:
        - Use df.dropna() to drop rows with null values
        - Or use df.na.fill() to fill with specific values
        """
        print(f"\nHandling missing values...")
        
        count_before = df.count()
        
        # Drop rows with any nulls (usually the first few rows of history due to window lags)
        df = df.na.drop()
        
        count_after = df.count()
        dropped = count_before - count_after
        print(f"   -> Dropped {dropped} rows (startup period for indicators).")
        
        return df

    def save_to_parquet(self, df, output_path):
        """
        Save processed data to Parquet format

        Args:
            df: Processed Spark DataFrame
            output_path: Path to save Parquet file

        TODO: Students - Save DataFrame as Parquet (2 points)

        HINTS:
        - Use df.write.parquet()
        - Use mode='overwrite' to replace existing files
        - Parquet is columnar format, faster than CSV
        """
        print(f"\nSaving to Parquet: {output_path}")

        # Clean existing
        if os.path.exists(output_path):
            shutil.rmtree(output_path)
            
        # Coalesce to 1 file (optimization for small data/8GB RAM)
        df.coalesce(1).write.mode("overwrite").parquet(output_path)
        print(f"✅ Saved processed data to {output_path}")


def run_preprocessing(spark=None):
    """
    Main function to run preprocessing pipeline

    TODO: Students - Complete this function

    STEPS:
    1. Create SparkPreprocessor instance
    2. Load CSV files
    3. Run preprocessing
    4. Save to Parquet
    5. Show sample data
    """
    print("="*60)
    print("PYSPARK DATA PREPROCESSING")
    print("="*60)


    # 1. Create Processor
    processor = SparkPreprocessor(spark)
    
    # 2. Load Data
    df = processor.load_csv_files()
    
    # 3. Feature Engineering
    df = processor.calculate_moving_averages(df)
    df = processor.calculate_rsi(df)
    df = processor.calculate_volatility(df)
    df = processor.calculate_returns(df)
    
    # 4. Handle Missing
    df = processor.handle_missing_values(df)
    
    # 5. Save
    processor.save_to_parquet(df, PARQUET_PATH)
    
    return df


if __name__ == "__main__":
    """
    Run this file to test your preprocessing

    Usage: python preprocessing/spark_preprocessor.py
    """
    # Check for Java
    if not os.environ.get("JAVA_HOME"):
        print("⚠️  WARNING: JAVA_HOME is not set. PySpark requires Java.")

    try:
        run_preprocessing()
    except Exception as e:
        print(f"\n❌ Preprocessing Failed: {e}")
        import traceback
        traceback.print_exc()