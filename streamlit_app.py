import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from stable_baselines3 import PPO
from env.trading_env import TradingEnv
from io import StringIO

st.set_page_config(page_title="🧠 Stock Market Simulator", layout="centered")

st.title("🧠 Stock Market Simulator (PPO Agent)")
st.markdown("Upload a stock CSV and simulate trading using a trained PPO model.")

# Upload CSV
uploaded_file = st.file_uploader("📤 Upload CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        # Read CSV safely
        stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
        df = pd.read_csv(stringio)

        # Validate CSV
        if df.empty or len(df.columns) == 0:
            st.error("❌ The uploaded CSV is empty or has no valid columns.")
        else:
            st.success("✅ File successfully uploaded!")

            # Display data
            st.subheader("📊 Raw Data")
            st.dataframe(df.head())

            # Ensure Close column is numeric
            df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
            df['Open'] = pd.to_numeric(df['Open'], errors='coerce')

            # Drop NaNs from conversions
            df.dropna(subset=['Close', 'Open'], inplace=True)

            # Add indicators if not present
            if 'MA10' not in df.columns:
                df['MA10'] = df['Close'].rolling(10).mean()
            if 'MA50' not in df.columns:
                df['MA50'] = df['Close'].rolling(50).mean()
            df['Price'] = df['Close']  # PPO env expects 'Price'

            # Drop rows with NaNs after rolling
            df.dropna(inplace=True)

            # Simulate
            st.subheader("🚀 Trading Simulation")

            env = TradingEnv(df.reset_index(drop=True))

            model_path = "ppo_tatasteel.zip"
            if os.path.exists(model_path):
                model = PPO.load(model_path)

                obs = env.reset()
                rewards = []
                prices = []
                values = []

                for _ in range(len(df) - 1):
                    action, _states = model.predict(obs)
                    obs, reward, done, info = env.step(action)
                    rewards.append(reward)
                    prices.append(info['price'])
                    values.append(info['value'])
                    if done:
                        break

                st.success("✅ Simulation Complete")

                # Visualize
                st.subheader("📈 Portfolio Value Over Time")
                fig, ax = plt.subplots()
                ax.plot(values)
                ax.set_xlabel("Time Step")
                ax.set_ylabel("Portfolio Value")
                ax.set_title("Portfolio Value Simulation")
                st.pyplot(fig)

                st.subheader("📉 Stock Price Over Time")
                fig2, ax2 = plt.subplots()
                ax2.plot(prices)
                ax2.set_xlabel("Time Step")
                ax2.set_ylabel("Price")
                ax2.set_title("Stock Price")
                st.pyplot(fig2)

            else:
                st.warning("⚠️ Trained model not found! Please train the PPO model and save it as `ppo_tatasteel.zip`.")
    except Exception as e:
        st.error(f"❌ Error reading file: {e}")
