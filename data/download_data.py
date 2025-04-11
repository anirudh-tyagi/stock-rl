import yfinance as yf
import pandas as pd
import os
import matplotlib.pyplot as plt

def download_tatasteel(start_date="2010-01-01", end_date="2025-04-01", interval="1d", save=True):
    ticker = "TATASTEEL.NS"
    df = yf.download(ticker, start=start_date, end=end_date, interval=interval)

    # Add technical indicators (10-day and 50-day moving averages)
    df["MA10"] = df["Close"].rolling(window=10).mean()
    df["MA50"] = df["Close"].rolling(window=50).mean()
    df.dropna(inplace=True)

    if save:
        os.makedirs("data", exist_ok=True)
        df.to_csv("data/tatasteel_daily.csv")
        print("✅ Daily data saved to data/tatasteel_daily.csv")

    return df

if __name__ == "__main__":
    df = download_tatasteel()
    print(df.head())

    # Optional: plot
    df[["Close", "MA10", "MA50"]].plot(figsize=(12,6))
    plt.title("Tata Steel Daily Stock Prices with MA10 and MA50")
    plt.xlabel("Date")
    plt.ylabel("Price (INR)")
    plt.grid(True)
    plt.show()
