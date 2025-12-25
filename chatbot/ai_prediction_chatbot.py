"""
AI Prediction Chatbot (Task 6)
Integrates Llama 3.2 (Ollama), Spark ML, and SQLite
Final Version: Includes full indicator table and professional formatting.
"""

import streamlit as st
import ollama
import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import io
import base64
import sys
import os
import re

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config.config import DB_PATH, DEFAULT_TICKERS, LLM_MODEL, SYSTEM_PROMPT
from ml_models.spark_gbt_forecaster import SparkGBTForecaster

# Page Config
st.set_page_config(page_title="AI Financial Assistant", page_icon="🤖", layout="wide")

class FinancialChatbot:
    def __init__(self):
        """Initialize Database and ML Models"""
        self.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        
        # Initialize Spark Forecaster (Cache in Session State to avoid reloading)
        if 'forecaster' not in st.session_state:
            try:
                # st.spinner("Loading AI Brain & ML Models...")
                forecaster = SparkGBTForecaster()
                forecaster.load_model()
                
                # Load data into memory for fast inference
                st.session_state.spark_data = forecaster.load_data()
                st.session_state.forecaster = forecaster
                # st.success("System Ready!")
            except Exception as e:
                st.error(f"Failed to load ML Model: {e}")

    def get_stock_data(self, ticker, days=30):
        """
        Query SQLite for historical data (Includes EXTRA indicators for table)
        """
        query = f"""
        SELECT date, close, rsi, volatility, daily_return, ma_7 
        FROM stock_data 
        WHERE ticker = '{ticker}' 
        ORDER BY date DESC 
        LIMIT {days}
        """
        df = pd.read_sql(query, self.conn)
        return df.sort_values('date')

    def get_prediction(self, ticker, num_days=7):
        """
        Generate predictions using Spark GBT model
        """
        if 'forecaster' not in st.session_state:
            return pd.DataFrame()
            
        return st.session_state.forecaster.predict_future(
            st.session_state.spark_data, ticker, num_days
        )

    def generate_prediction_graph(self, hist_df, pred_df, ticker):
        """
        Create a combined chart: Historical (Blue) + Predicted (Red)
        """
        plt.figure(figsize=(10, 5))
        
        # Plot Historical Data
        plt.plot(pd.to_datetime(hist_df['date']), hist_df['close'], 
                 label='Historical (Last 30 Days)', color='#1f77b4', linewidth=2)
        
        # Plot Prediction Data
        # Connect the last historical point to the first prediction point
        last_hist_date = pd.to_datetime(hist_df['date'].iloc[-1])
        last_hist_price = hist_df['close'].iloc[-1]
        
        pred_dates = [last_hist_date] + pd.to_datetime(pred_df['Date']).tolist()
        pred_prices = [last_hist_price] + pred_df['Predicted_Close'].tolist()
        
        plt.plot(pred_dates, pred_prices, 
                 label=f'Prediction (Next {len(pred_df)} Days)', 
                 color='#d62728', linestyle='--', linewidth=2, marker='o')

        plt.title(f"{ticker} Price Prediction Analysis")
        plt.xlabel("Date")
        plt.ylabel("Price ($)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.xticks(rotation=45)
        
        # Convert to base64 string
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def detect_intent(self, prompt):
        """
        Detect User Intent (Prediction vs Data vs Chat)
        """
        prompt = prompt.lower()
        
        # Extract Ticker (e.g., AAPL, MSFT)
        ticker_match = next((t for t in DEFAULT_TICKERS if t.lower() in prompt), None)
        
        # Extract Number of Days (default 7)
        days_match = re.search(r'(\d+)\s*days?', prompt)
        num_days = int(days_match.group(1)) if days_match else 7
        
        if "predict" in prompt or "forecast" in prompt:
            return "prediction", ticker_match, num_days
        elif "show" in prompt or "price" in prompt or "data" in prompt or "tell me about" in prompt:
            return "data", ticker_match, 30
        else:
            return "chat", None, None

    def get_llm_response(self, prompt, context=""):
        """
        Query Ollama (Llama 3.2) for general questions
        """
        full_prompt = f"{SYSTEM_PROMPT}\n\nContext Info: {context}\n\nUser Question: {prompt}"
        
        try:
            response = ollama.chat(model=LLM_MODEL, messages=[
                {'role': 'user', 'content': full_prompt}
            ])
            return response['message']['content']
        except Exception as e:
            return f"⚠️ Error connecting to AI Brain: {str(e)}. Please ensure Docker and Ollama are running."

    def run(self):
        """Main Chatbot Loop"""
        st.title("🤖 AI Financial Analyst")
        st.markdown("Ask me to **predict AAPL**, **tell me about TSLA**, or explain **RSI**!")

        # Initialize Chat History
        if "messages" not in st.session_state:
            st.session_state.messages = []

        # Display Chat History
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
                if "image" in message:
                    st.image(base64.b64decode(message["image"]))

        # Handle User Input
        if prompt := st.chat_input("Ex: Predict AAPL next 7 days"):
            # 1. Display User Message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # 2. Process Intent & Generate Response
            with st.chat_message("assistant"):
                message_placeholder = st.empty()
                message_placeholder.markdown("Thinking...")
                
                intent, ticker, num_days = self.detect_intent(prompt)
                
                response_text = ""
                graph_image = None
                
                # --- Scenario A: Prediction ---
                if intent == "prediction":
                    if ticker:
                        pred_df = self.get_prediction(ticker, num_days)
                        hist_df = self.get_stock_data(ticker, days=30)
                        
                        if not pred_df.empty and not hist_df.empty:
                            last_price = pred_df['Predicted_Close'].iloc[-1]
                            response_text = f"### 🔮 Prediction for {ticker}\n"
                            response_text += f"I forecast **{ticker}** to reach **${last_price:.2f}** in {num_days} days.\n\n"
                            response_text += pred_df.to_markdown(index=False)
                            
                            # Generate Graph
                            graph_image = self.generate_prediction_graph(hist_df, pred_df, ticker)
                        else:
                            response_text = f"Sorry, I couldn't generate a prediction for {ticker}."
                    else:
                        response_text = "Please specify a ticker symbol (e.g., AAPL, MSFT)."

                # --- Scenario B: Data Retrieval (UPDATED FOR TABLE) ---
                elif intent == "data":
                    if ticker:
                        # Fetch 30 days of data
                        df = self.get_stock_data(ticker, days=30)
                        
                        if not df.empty:
                            # 1. Prepare Table (Last 10 Days)
                            last_10_df = df.tail(10).sort_values(by='date', ascending=False)
                            
                            # Create display copy with specific columns
                            display_df = last_10_df[['date', 'close', 'daily_return', 'rsi', 'volatility', 'ma_7']].copy()
                            
                            # Format columns
                            display_df['close'] = display_df['close'].apply(lambda x: f"${x:.2f}")
                            display_df['ma_7'] = display_df['ma_7'].apply(lambda x: f"${x:.2f}")
                            display_df['rsi'] = display_df['rsi'].apply(lambda x: f"{x:.2f}")
                            display_df['volatility'] = display_df['volatility'].apply(lambda x: f"{x:.4f}")
                            display_df['daily_return'] = display_df['daily_return'].apply(lambda x: f"{x*100:.2f}%")
                            
                            # Rename for UI
                            display_df.columns = ['Date', 'Close', 'Return %', 'RSI', 'Volatility', '7-Day MA']
                            
                            response_text = f"### 📊 Recent Data for {ticker}\n"
                            response_text += "Here are the latest 10 days of market indicators:\n\n"
                            response_text += display_df.to_markdown(index=False)
                            
                            # 2. Generate Graph
                            plt.figure(figsize=(10, 4))
                            plt.plot(pd.to_datetime(df['date']), df['close'], label='Close Price', linewidth=2)
                            plt.plot(pd.to_datetime(df['date']), df['ma_7'], label='7-Day MA', linestyle='--', alpha=0.7)
                            plt.title(f"{ticker} 30-Day Trend")
                            plt.xlabel("Date")
                            plt.ylabel("Price")
                            plt.grid(True, alpha=0.3)
                            plt.legend()
                            
                            buf = io.BytesIO()
                            plt.savefig(buf, format='png', bbox_inches='tight')
                            plt.close()
                            graph_image = base64.b64encode(buf.getvalue()).decode('utf-8')
                            
                        else:
                            response_text = f"No data found for {ticker}."
                    else:
                        response_text = "Which stock would you like to see?"

                # --- Scenario C: General Chat (LLM) ---
                else:
                    response_text = self.get_llm_response(prompt)

                # 3. Display Final Response
                message_placeholder.markdown(response_text)
                if graph_image:
                    st.image(base64.b64decode(graph_image))
                
                # 4. Save to History
                msg_data = {"role": "assistant", "content": response_text}
                if graph_image:
                    msg_data["image"] = graph_image
                st.session_state.messages.append(msg_data)

if __name__ == "__main__":
    bot = FinancialChatbot()
    bot.run()