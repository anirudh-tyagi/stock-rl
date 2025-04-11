import pandas as pd
import torch
from env.trading_env import TradingEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Check for GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("🔥 Using device:", device)

# Load data
df = pd.read_csv("data/tatasteel_daily.csv")
df.columns = df.columns.str.strip()

# Keep relevant columns only
required_cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'MA10', 'MA50']
df = df[required_cols].dropna().astype('float32')

# Create trading environment
env = DummyVecEnv([lambda: TradingEnv(df)])

# Initialize PPO agent
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./tensorboard_logs/",
    device=device
)

# Train the model
model.learn(total_timesteps=100_000)

# Save the trained model
model.save("ppo_tatasteel")
print("✅ Model saved as ppo_tatasteel.zip")
