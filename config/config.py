"""
Configuration settings for the AI-Powered Financial Analysis Platform

STUDENT TASK:
- Review all configuration settings
- Modify paths if needed for your system
- Add any additional configuration you need
- DO NOT hardcode sensitive information (API keys, passwords)
"""

import os
from pathlib import Path
# ============================================================================
# PROJECT PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).parent.parent.absolute()

# Data Directories
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
STOCK_DATA_DIR = os.path.join(DATA_DIR, "stock_data")
MODELS_DIR = os.path.join(DATA_DIR, "models")
LOGS_DIR = os.path.join(DATA_DIR, "logs")

# File Paths
DB_PATH = os.path.join(DATA_DIR, "financial_data.db")
PARQUET_PATH = os.path.join(DATA_DIR, "processed_stocks.parquet")
GBT_MODEL_PATH = os.path.join(MODELS_DIR, "gbt_forecaster")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "investment_classifier")

# Ensure directories exist
os.makedirs(STOCK_DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# ============================================================================
# DATA COLLECTION SETTINGS
# ============================================================================
# Tickers to download (Task 1)
DEFAULT_TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']

# Date range for historical data
START_DATE = "2020-01-01"
DATE_FORMAT = "%Y-%m-%d"

# ============================================================================
# SPARK & PROCESSING SETTINGS
# ============================================================================
APP_NAME = "FinancialAnalysisPlatform"
MASTER_URL = "local[*]"  # Use all available cores

# Memory Settings (Adjust based on your RAM as per SETUP.md)
# For 16GB RAM systems: "4g", for 8GB RAM systems: "2g"
SPARK_DRIVER_MEMORY = "2g"
SPARK_EXECUTOR_MEMORY = "2g"

# Feature Engineering Parameters
MA_WINDOWS = [7, 30, 90]        # Moving Average windows
RSI_PERIOD = 14                 # RSI calculation period
VOLATILITY_WINDOW = 30          # Rolling standard deviation window
LOOKBACK_DAYS = 30              # Days of history to use for prediction (lags)
PREDICT_DAYS = 7                # Days into the future to predict

# ============================================================================
# MACHINE LEARNING HYPERPARAMETERS
# ============================================================================
# Task 4: GBT Regressor Settings
GBT_PARAMS = {
    "maxIter": 100,             # Number of trees
    "maxDepth": 6,              # Depth of trees
    "stepSize": 0.1,            # Learning rate
    "subsamplingRate": 0.8,     # Prevent overfitting
    "seed": 42
}

# Task 5: Investment Classifier Settings
RF_PARAMS = {
    "numTrees": 100,
    "maxDepth": 10,
    "seed": 42
}

# Data Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1

# ============================================================================
# LLM & CHATBOT SETTINGS
# ============================================================================
# Ollama Configuration
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL = "llama3.2"  # or "llama3.1" if you downloaded the larger one

# Chatbot System Prompt
SYSTEM_PROMPT = """
You are a financial analysis assistant. 
You answer questions about stock trends, technical indicators, and investment concepts.
Use the context provided to answer specific questions about stock predictions.
Do not provide financial advice. Always include a disclaimer.
"""

# ============================================================================
# DASHBOARD SETTINGS
# ============================================================================
DASHBOARD_TITLE = "AI Financial Analysis Platform"
DASHBOARD_LAYOUT = "wide"