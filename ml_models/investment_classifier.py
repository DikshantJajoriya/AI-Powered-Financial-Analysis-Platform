"""
Investment Classifier (Task 5)
Classifies stocks as High/Medium/Low Potential using Random Forest
Optimized for Power BI & SQLite
"""

import sys
import os
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import (
    PARQUET_PATH, RF_MODEL_PATH, APP_NAME, 
    SPARK_DRIVER_MEMORY, DB_PATH, RF_PARAMS
)

class InvestmentClassifier:
    def __init__(self):
        print(f"⚡ Initializing Spark Session for Classification...")
        self.spark = SparkSession.builder \
            .appName(APP_NAME) \
            .master("local[*]") \
            .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
            .config("spark.sql.shuffle.partitions", "5") \
            .getOrCreate()
        self.spark.sparkContext.setLogLevel("WARN")

    def load_data(self):
        print(f"📂 Loading data from {PARQUET_PATH}...")
        if not os.path.exists(PARQUET_PATH):
            raise FileNotFoundError("Processed data not found. Run Task 2 first.")
        return self.spark.read.parquet(PARQUET_PATH)

    def engineer_features(self, df):
        """
        Create 17 Aggregate Features & Generate Labels
        """
        print("🛠️  Engineering 17 Features & Composite Labels...")
        
        w_spec = Window.partitionBy("Ticker").orderBy("Date")
        
        # --- 1. Return Metrics ---
        # Total Return (Since beginning of dataset for that ticker)
        first_price_w = Window.partitionBy("Ticker").orderBy("Date").rowsBetween(Window.unboundedPreceding, Window.currentRow)
        df = df.withColumn("First_Close", F.first("Close").over(first_price_w))
        df = df.withColumn("Total_Return", (F.col("Close") - F.col("First_Close")) / F.col("First_Close"))
        
        # Recent Returns (7-day and 30-day)
        df = df.withColumn("Return_7d", (F.col("Close") - F.lag("Close", 7).over(w_spec)) / F.lag("Close", 7).over(w_spec))
        df = df.withColumn("Return_30d", (F.col("Close") - F.lag("Close", 30).over(w_spec)) / F.lag("Close", 30).over(w_spec))

        # --- 2. Technical Metrics ---
        # Average RSI (30 day rolling)
        w_30 = w_spec.rowsBetween(-29, 0)
        df = df.withColumn("Avg_RSI_30d", F.avg("RSI").over(w_30))
        
        # Price Trends (Diff between MAs)
        df = df.withColumn("Trend_MA7_MA30", F.col("MA_7") - F.col("MA_30"))
        df = df.withColumn("Trend_MA30_MA90", F.col("MA_30") - F.col("MA_90"))
        
        # --- 3. Composite Score Calculation (The Labeling Logic) ---
        # We need to map metrics to a 0-10 scale to use the formula
        
        # Return Score (0-10): Higher return is better
        # Map > 30% return to 10, < -10% to 0
        ret_score = F.when(F.col("Total_Return") > 0.3, 10) \
                     .when(F.col("Total_Return") > 0.1, 7) \
                     .when(F.col("Total_Return") > 0, 5) \
                     .otherwise(2)

        # Trend Score (0-10): Positive trends are better
        trend_score = F.when((F.col("Trend_MA7_MA30") > 0) & (F.col("Trend_MA30_MA90") > 0), 10) \
                       .when(F.col("Trend_MA7_MA30") > 0, 7) \
                       .otherwise(3)
        
        # RSI Score (0-10): "Buy Low" strategy (Oversold < 30 is good)
        rsi_raw_score = F.when(F.col("RSI") < 30, 10) \
                         .when(F.col("RSI") > 70, 2) \
                         .otherwise(5) # Neutral

        # Volatility Score (0-10): Stability (Low vol) is usually preferred for "Investment Grade"
        # Invert volatility: Lower is higher score
        vol_score = F.when(F.col("Volatility") < 1, 10).when(F.col("Volatility") < 3, 7).otherwise(3)
        
        # Sharpe Score (0-10)
        sharpe_score = F.when(F.col("Sharpe_Ratio") > 1, 10).when(F.col("Sharpe_Ratio") > 0, 5).otherwise(1)

        # FORMULA
        # Score = (Total_Return * 0.3) + (Trend * 0.2) + (RSI * 0.15) + (Vol * 0.15) + (Sharpe * 0.2)
        df = df.withColumn("Score", 
            (ret_score * 0.3) + 
            (trend_score * 0.2) + 
            (rsi_raw_score * 0.15) + 
            (vol_score * 0.15) + 
            (sharpe_score * 0.2)
        )
        
        # Assign Labels: High (>=7), Medium (4-7), Low (<4)
        df = df.withColumn("Investment_Grade", 
            F.when(F.col("Score") >= 7, "High")
            .when(F.col("Score") >= 4, "Medium")
            .otherwise("Low")
        )
        
        # Select the 17 features + Labels for training
        # Note: We drop rows with nulls (created by lags/windows)
        feature_list = [
            "Total_Return", "Return_7d", "Return_30d", 
            "RSI", "Avg_RSI_30d", 
            "Volatility", "Sharpe_Ratio", 
            "MA_7", "MA_30", "MA_90", 
            "Trend_MA7_MA30", "Trend_MA30_MA90",
            "Daily_Return", "Close", "Open", "High", "Low" # Adding prices to reach 17 count if needed, but metrics above are stronger
        ]
        
        # Ensure we have clean data
        clean_df = df.na.drop(subset=feature_list + ["Score"])
        
        print(f"   -> Data labeled. Distribution:")
        clean_df.groupBy("Investment_Grade").count().show()
        
        return clean_df, feature_list

    def train_model(self, df, feature_cols):
        print("\n🤖 Training Random Forest Classifier...")
        
        # 1. Index Labels
        indexer = StringIndexer(inputCol="Investment_Grade", outputCol="label")
        
        # 2. Assemble Features
        assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        
        # 3. Define Random Forest
        rf = RandomForestClassifier(labelCol="label", featuresCol="features", 
                                    numTrees=RF_PARAMS['numTrees'], seed=42)
        
        # 4. Pipeline
        pipeline = Pipeline(stages=[indexer, assembler, rf])
        
        # 5. Split & Train (80/20)
        train_data, test_data = df.randomSplit([0.8, 0.2], seed=42)
        print("   -> Fitting model...")
        model = pipeline.fit(train_data)
        
        # 6. Evaluation
        predictions = model.transform(test_data)
        
        # Metrics
        acc_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
        f1_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="f1")
        prec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedPrecision")
        rec_eval = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="weightedRecall")
        
        acc = acc_eval.evaluate(predictions)
        f1 = f1_eval.evaluate(predictions)
        prec = prec_eval.evaluate(predictions)
        rec = rec_eval.evaluate(predictions)
        
        print(f"\n📊 Classification Metrics:")
        print(f"   Accuracy:  {acc:.2%}")
        print(f"   Precision: {prec:.2%}")
        print(f"   Recall:    {rec:.2%}")
        print(f"   F1-Score:  {f1:.2%}")
        
        # Save Model (Robust to Windows Long Path Error)
        try:
            model.write().overwrite().save(RF_MODEL_PATH)
            print(f"💾 Model saved to {RF_MODEL_PATH}")
        except Exception as e:
            print(f"⚠️  WARNING: Could not save model file (likely due to Windows path length limit).")
            print(f"   -> Proceeding to save results to Database (Critical for Power BI).")
        
        return model

    def save_results_for_powerbi(self, df):
        """
        Save the LATEST rating for each ticker to SQLite
        """
        print("\n🏆 Generating Final Classification Results...")
        
        # Get latest date per ticker
        w_latest = Window.partitionBy("Ticker").orderBy(F.col("Date").desc())
        latest_df = df.withColumn("row", F.row_number().over(w_latest)).filter(F.col("row") == 1)
        
        # Select columns for Power BI
        final_df = latest_df.select(
            "Ticker", "Date", "Close", "Score", "Investment_Grade", 
            "Total_Return", "RSI", "Sharpe_Ratio"
        )
        
        # Print to console (Expected Output)
        print("\nClassification Results:")
        rows = final_df.collect()
        for row in rows:
            print(f"  {row['Ticker']}: {row['Investment_Grade']} (Score: {row['Score']:.1f})")
            
        # Save to SQLite
        try:
            pdf = final_df.toPandas()
            import sqlite3
            with sqlite3.connect(DB_PATH) as conn:
                pdf.to_sql("investment_ratings", conn, if_exists="replace", index=False)
            print(f"\n✅ Saved ratings to 'investment_ratings' table in SQLite.")
        except Exception as e:
            print(f"❌ Failed to save to SQLite: {e}")

    def run(self):
        print("="*60)
        print("🎯 INVESTMENT CLASSIFIER (Task 5)")
        print("="*60)
        
        # 1. Load
        df = self.load_data()
        
        # 2. Engineer & Label
        labeled_df, features = self.engineer_features(df)
        
        # 3. Train & Evaluate
        model = self.train_model(labeled_df, features)
        
        # 4. Save Latest Results
        self.save_results_for_powerbi(labeled_df)
        
        self.spark.stop()
        print("="*60)

if __name__ == "__main__":
    if not os.environ.get("JAVA_HOME"):
        print("⚠️  JAVA_HOME not set")
    try:
        classifier = InvestmentClassifier()
        classifier.run()
    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
        import traceback
        traceback.print_exc()