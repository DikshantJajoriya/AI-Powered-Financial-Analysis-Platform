"""
Main Pipeline Orchestrator - AI Financial Analysis Platform

USAGE:
    python main.py

Then select from menu:
1. Data Collection
2. Preprocessing
3. Database Setup
4. Train ML Models
5. Run Chatbot
6. Run Complete Pipeline
"""

import os
import sys
import time
from pyspark.sql import SparkSession

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Project Modules
try:
    from config.config import APP_NAME, SPARK_DRIVER_MEMORY
    from data_collection.stock_downloader import run_data_collection
    from preprocessing.spark_preprocessor import run_preprocessing
    from sql_interface.database_manager import run_database_setup
    from ml_models.spark_gbt_forecaster import SparkGBTForecaster
    from ml_models.investment_classifier import InvestmentClassifier
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("Ensure you are running 'python main.py' from the project root.")
    sys.exit(1)


def initialize_spark():
    """
    Initialize a shared Spark session for the pipeline
    """
    print(f"\n⚡ Initializing Master Spark Session ({APP_NAME})...")
    spark = SparkSession.builder \
        .appName(APP_NAME) \
        .master("local[*]") \
        .config("spark.driver.memory", SPARK_DRIVER_MEMORY) \
        .config("spark.sql.shuffle.partitions", "5") \
        .getOrCreate()
    spark.sparkContext.setLogLevel("ERROR")
    return spark


def train_ml_models():
    """Run both GBT Forecaster and Investment Classifier"""
    print("\n" + "="*40)
    print("🧠 TRAINING MACHINE LEARNING MODELS")
    print("="*40)
    
    # 1. Run GBT Forecaster
    print("\n--- Step 1: Training Time Series Forecaster (GBT) ---")
    try:
        forecaster = SparkGBTForecaster()
        forecaster.run()
    except Exception as e:
        print(f"❌ GBT Training Failed: {e}")

    # 2. Run Investment Classifier
    print("\n--- Step 2: Training Investment Classifier (Random Forest) ---")
    try:
        classifier = InvestmentClassifier()
        classifier.run()
    except Exception as e:
        print(f"❌ Classifier Training Failed: {e}")


def run_chatbot():
    """Launch the CLI Chatbot"""
    print("\n💬 Launching Financial Chatbot...")
    # Assumes chatbot file is at chatbot/financial_chatbot.py
    chatbot_path = os.path.join("chatbot", "financial_chatbot.py")
    if os.path.exists(chatbot_path):
        os.system(f"python {chatbot_path}")
    else:
        print(f"❌ Chatbot file not found at: {chatbot_path}")


def main():
    """Main Menu Loop"""
    spark = None  # Lazy init

    while True:
        print("\n" + "="*50)
        print("   🚀 AI FINANCIAL ANALYSIS PLATFORM - MAIN MENU")
        print("="*50)
        print("1. 📥 Data Collection (Download Yahoo Finance Data)")
        print("2. 🧹 Data Preprocessing (Spark Feature Engineering)")
        print("3. 🗄️  Database Setup (Load Data to SQLite)")
        print("4. 🤖 Train ML Models (Forecasting & Classification)")
        print("5. 💬 Run Chatbot (Ollama/LLM)")
        print("6. ⚡ Run COMPLETE PIPELINE (Steps 1-4)")
        print("0. ❌ Exit")
        print("="*50)

        choice = input("👉 Enter your choice (0-6): ").strip()

        if choice == '1':
            run_data_collection()

        elif choice == '2':
            if not spark: spark = initialize_spark()
            run_preprocessing(spark)

        elif choice == '3':
            run_database_setup()

        elif choice == '4':
            train_ml_models()

        elif choice == '5':
            run_chatbot()

        elif choice == '6':
            print("\n🚀 STARTING FULL PIPELINE EXECUTION...")
            start_time = time.time()
            
            run_data_collection()
            
            if not spark: spark = initialize_spark()
            run_preprocessing(spark)
            
            run_database_setup()
            train_ml_models()
            
            elapsed = round((time.time() - start_time) / 60, 2)
            print(f"\n✅ FULL PIPELINE COMPLETE in {elapsed} minutes.")

        elif choice == '0':
            print("\n👋 Exiting...")
            if spark: spark.stop()
            sys.exit(0)

        else:
            print("\n❌ Invalid choice. Please try again.")
            
        input("\nPress Enter to continue...")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Interrupted by user")
    except Exception as e:
        print(f"\n❌ Unexpected Error: {e}")
        import traceback
        traceback.print_exc()