import pandas as pd
import torch
import matplotlib.pyplot as plt
from env.trading_env import TradingEnv
from stable_baselines3 import PPO

# Load and prepare data
df = pd.read_csv("data/tatasteel_daily.csv")
df.columns = df.columns.str.strip()
df = df[['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50']].dropna().astype('float32')

# Create environment
env = TradingEnv(df)

# Load trained model
model = PPO.load("ppo_tatasteel", device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))

# Run the model
obs = env.reset()
done = False
portfolio_values = []

while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    portfolio_values.append(env.total_value)

# Plot results
plt.figure(figsize=(12, 6))
plt.plot(portfolio_values, label="Portfolio Value", color='dodgerblue')
plt.title("📈 PPO Agent Portfolio Value Over Time")
plt.xlabel("Trading Steps")
plt.ylabel("Portfolio Value ₹")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
