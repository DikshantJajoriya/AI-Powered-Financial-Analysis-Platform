"""
Spark MLlib Gradient Boosted Trees for Time Series Forecasting

STUDENT TASK (35 points):
Implement time series forecasting using Spark MLlib GBT Regressor

LEARNING OBJECTIVES:
- Create lagged features for time series
- Train Gradient Boosted Trees model
- Make multi-step predictions
- Evaluate model performance

EXPECTED OUTPUT:
- Trained model saved to data/models/spark_gbt_forecaster/
- Test R² Score: 0.90+ (90%+ accuracy)
- Test RMSE: $25-40
- Mean % Error: < 10%
"""

import sys
import os
import shutil
import pandas as pd
import numpy as np
from datetime import timedelta, datetime
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import abs, col, avg, lag, lit, when, max as spark_max, rand
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline, PipelineModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.config import (
    PARQUET_PATH, GBT_MODEL_PATH, APP_NAME, 
    SPARK_DRIVER_MEMORY, PREDICT_DAYS, LOOKBACK_DAYS,
    DB_PATH, GBT_PARAMS
)


class SparkGBTForecaster:
    """
    Time Series Forecasting using Spark MLlib Gradient Boosted Trees

    TODO: Students - Complete all methods marked with TODO
    """

    def __init__(self):
        """
        Initialize forecaster

        Args:
            spark: SparkSession
            lookback_days: Number of historical days to use as features
            forecast_days: Number of days to forecast

        TODO: Complete initialization
        """
        print(f"⚡ Initializing Spark Session for ML...")
        self.spark = SparkSession.builder \
            .appName(APP_NAME) \
            .master("local[*]") \
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
            .config("spark.sql.shuffle.partitions", "5") \
            .config("spark.cleaner.referenceTracking.cleanCheckpoints", "true") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")
        self.model = None

    def load_data(self):
        """Load processed Parquet data"""
        print(f"📂 Loading data from {PARQUET_PATH}...")
        if not os.path.exists(PARQUET_PATH):
            raise FileNotFoundError("Processed data not found. Run Task 2 first.")
        df = self.spark.read.parquet(PARQUET_PATH)
        return df

    def create_lagged_features(self, df):
        """
        Create lagged features for time series forecasting

        Args:
            df: Spark DataFrame with columns [Ticker, Date, Open, High, Low, Close, Volume]

        Returns:
            DataFrame with lagged features

        TODO: Students - Implement this function (15 points)

        TASK:
        Create 150 lagged features:
        - Close_lag_1 to Close_lag_30 (30 features)
        - Open_lag_1 to Open_lag_30 (30 features)
        - High_lag_1 to High_lag_30 (30 features)
        - Low_lag_1 to Low_lag_30 (30 features)
        - Volume_lag_1 to Volume_lag_30 (30 features)

        IMPORTANT: Create features in BATCHES to avoid StackOverflowError!

        HINTS:
        1. Use Window.partitionBy('Ticker').orderBy('Date')
        2. Use lag(col, N) to get value N days ago
        3. Create in batches: 1-10, 11-20, 21-30
        4. Use .cache() and .count() after each batch

        EXAMPLE:
            window = Window.partitionBy('Ticker').orderBy('Date')
            for lag_days in range(1, 11):  # Batch 1: lag 1-10
                df = df.withColumn(f'Close_lag_{lag_days}',
                                   lag(col('Close'), lag_days).over(window))
            df = df.cache()
            df.count()  # Force computation

            # Repeat for batches 11-20 and 21-30
        """
        # Only print if we are processing a large amount of data to avoid spamming chatbot logs
        if df.count() > 2000:
            print(f"🛠️  Engineering 150 Lagged Features ({LOOKBACK_DAYS} days history)...")
        
        window_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        # 1. Create Target (Future Close Price)
        df = df.withColumn("Target", lag(col("Close"), -PREDICT_DAYS).over(window_spec))
        
        # Columns to lag
        cols_to_lag = ['Close', 'Open', 'High', 'Low', 'Volume']
        feature_cols = ["RSI", "Volatility", "MA_7", "MA_30", "MA_90"] 
        
        # 2. Loop in batches to prevent memory crash
        batch_size = 5
        
        for i in range(1, LOOKBACK_DAYS + 1, batch_size):
            end_lag = min(i + batch_size, LOOKBACK_DAYS + 1)
            # print(f"   -> Processing lags {i} to {end_lag-1}...")
            
            for lag_day in range(i, end_lag):
                for col_name in cols_to_lag:
                    feat_name = f"{col_name}_lag_{lag_day}"
                    df = df.withColumn(feat_name, lag(col(col_name), lag_day).over(window_spec))
                    feature_cols.append(feat_name)
            
            # CRITICAL: Cut lineage to free memory
            df = df.localCheckpoint()

        
        # print(f"   -> Feature engineering complete. Total features: {len(feature_cols)}")
        return df, feature_cols

    def train_model(self, df, feature_cols):
        """
        Train the GBT model

        Args:
            spark_df: Spark DataFrame with features

        Returns:
            dict: Training metrics

        TODO: Students - Implement training (10 points)

        STEPS:
        1. Create lagged features
        2. Rename 'target' to 'label'
        3. Split data: 80% train, 10% val, 10% test
        4. Create VectorAssembler
        5. Create GBTRegressor
        6. Create Pipeline
        7. Train model
        8. Evaluate on train/val/test sets

        HINTS:
        - Use VectorAssembler to combine features into single vector
        - Use GBTRegressor with hyperparameters from config
        - Use RegressionEvaluator for metrics (RMSE, R², MAE)
        """
        print("\n🤖 Training GBT Model...")
        
        # Drop nulls for training
        train_df = df.na.drop(subset=["Target"] + feature_cols)
        
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        
        # 1. Strict Split: 80% Train, 10% Validation, 10% Test
        train_data, val_data, test_data = train_df.randomSplit([0.8, 0.1, 0.1], seed=42)
        
        train_data.cache() 
        print(f"   -> Data Split: Train={train_data.count()}, Val={val_data.count()}, Test={test_data.count()}")

        # Define GBT
        gbt = GBTRegressor(featuresCol="features", labelCol="Target", 
                           maxIter=GBT_PARAMS['maxIter'], 
                           maxDepth=GBT_PARAMS['maxDepth'],
                           seed=GBT_PARAMS['seed'])
        
        pipeline = Pipeline(stages=[assembler, gbt])
        
        print("   -> Fitting model (this may take 2-5 minutes)...")
        model = pipeline.fit(train_data)
        
        # 2. Evaluate on Test Set
        print("   -> Evaluating model on Test Set...")
        predictions = model.transform(test_data)
        
        rmse_evaluator = RegressionEvaluator(labelCol="Target", predictionCol="prediction", metricName="rmse")
        rmse = rmse_evaluator.evaluate(predictions)
        
        r2_evaluator = RegressionEvaluator(labelCol="Target", predictionCol="prediction", metricName="r2")
        r2 = r2_evaluator.evaluate(predictions)
        
        mae_evaluator = RegressionEvaluator(labelCol="Target", predictionCol="prediction", metricName="mae")
        mae = mae_evaluator.evaluate(predictions)
        
        mape_df = predictions.withColumn("APE", abs((col("Target") - col("prediction")) / col("Target")) * 100)
        mape = mape_df.select(avg("APE")).collect()[0][0]
        
        print(f"\n📊 Model Performance (Test Set):")
        print(f"   RMSE:          ${rmse:.2f}")
        print(f"   MAE:           ${mae:.2f}")
        print(f"   R²:            {r2:.4f}")
        print(f"   Mean % Error:  {mape:.2f}%")
        
        self.model = model
        return model
    

    def predict_future(self, df, ticker, num_days=7):
        """
        Predict future stock prices

        Args:
            df: Spark DataFrame with historical data
            ticker: Stock ticker
            num_days: Number of days to predict

        Returns:
            Pandas DataFrame with predictions

        TODO: Students - Implement prediction (10 points)

        STEPS:
        1. Filter data for specific ticker
        2. Create lagged features
        3. Get most recent features
        4. Make prediction
        5. Add realistic variation between days

        IMPORTANT: Add 1% random variation to avoid same predictions
        """
        print(f"\n🔮 Generating {num_days}-day forecast for {ticker}...")

        # 1. OPTIMIZATION: Filter for specific ticker FIRST.
        ticker_df = df.filter(col("Ticker") == ticker)
        
        if ticker_df.count() == 0:
            return pd.DataFrame()

        # 2. RE-ENGINEER FEATURES
        df_with_features, feature_cols = self.create_lagged_features(ticker_df)
        
        # 3. Assemble Features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="skip")
        data_with_vectors = assembler.transform(df_with_features)
        
        # 4. Get the most recent row (Current State)
        latest_row = data_with_vectors.orderBy(col("Date").desc()).limit(1).collect()
        
        if not latest_row:
            print(f"❌ No data found for {ticker}")
            return pd.DataFrame()
            
        latest_data = latest_row[0]
        current_price = latest_data['Close']
        last_date = latest_data['Date']
        
        # 5. Make Base Prediction (T+7 Trading Days)
        features_vector = latest_data['features']
        
        if len(features_vector) < 155: 
             print(f"❌ Feature Mismatch: Expected 155, got {len(features_vector)}")
             return pd.DataFrame()

        predicted_price_7_days = self.model.stages[-1].predict(features_vector)
        
        # 6. Interpolate and add variation (SKIPPING WEEKENDS)
        future_dates = []
        future_prices = []
        
        # Calculate daily step based on trading days
        price_step = (predicted_price_7_days - current_price) / 7
        current_date_obj = pd.to_datetime(last_date)
        
        days_generated = 0
        days_ahead = 0
        
        # Keep looping until we have generated 'num_days' of valid trading days
        while days_generated < num_days:
            days_ahead += 1
            next_date = current_date_obj + timedelta(days=days_ahead)
            
            # Skip Saturday (5) and Sunday (6)
            if next_date.weekday() >= 5:
                continue
            
            # If it's a weekday, keep it
            days_generated += 1
            
            # Base price calculation
            base_price = current_price + (price_step * days_generated)
            noise = np.random.normal(0, base_price * 0.01) # 1% random variation
            final_price = base_price + noise
            
            future_dates.append(next_date.strftime('%Y-%m-%d'))
            future_prices.append(round(final_price, 2))
            
        return pd.DataFrame({
            'Ticker': [ticker] * num_days,
            'Date': future_dates,
            'Predicted_Close': future_prices
        })

    def save_model(self):
        """Save trained model"""
        if self.model:
            print(f"💾 Saving model to {GBT_MODEL_PATH}...")
            if os.path.exists(GBT_MODEL_PATH):
                shutil.rmtree(GBT_MODEL_PATH)
            self.model.write().save(GBT_MODEL_PATH)

    def load_model(self):
        """Load trained model"""
        print(f"📂 Loading model from {GBT_MODEL_PATH}...")
        if not os.path.exists(GBT_MODEL_PATH):
            print("❌ Model file not found. Please train first.")
            return None
        self.model = PipelineModel.load(GBT_MODEL_PATH)
        print("✅ Model loaded successfully.")
        return self.model
    
    def run(self):
        print("="*60)
        print("🚀 SPARK GBT FORECASTER (High-Dimension)")
        print("="*60)
        
        df = self.load_data()
        
        # Train
        df_features, feature_cols = self.create_lagged_features(df)
        self.train_model(df_features, feature_cols)
        self.save_model()
        self.save_predictions_for_powerbi(df_features, self.model, feature_cols)
        
        # Test Chatbot Prediction
        print("\n--- Testing Chatbot Function ---")
        pred_df = self.predict_future(df, "AAPL", 7)
        print(pred_df)
        
        self.spark.stop()
        print("="*60)


if __name__ == "__main__":
    """
    Test your forecaster

    Usage: python ml_models/spark_gbt_forecaster.py
    """
    if not os.environ.get("JAVA_HOME"):
        print("⚠️  JAVA_HOME not set")
    try:
        forecaster = SparkGBTForecaster()
        forecaster.run()
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        import traceback
        traceback.print_exc()