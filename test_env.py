import pandas as pd
from env.trading_env import TradingEnv

# Load CSV and drop non-numeric columns (like 'Date', 'Unnamed', etc.)
df = pd.read_csv("data/tatasteel_daily.csv")

# Drop any non-numeric columns
df = df.select_dtypes(include=['number'])

# Drop any rows with NaN (e.g., early moving averages)
df.dropna(inplace=True)

# Convert to float32
df = df.astype('float32')

# Test environment
env = TradingEnv(df)
obs = env.reset()
done = False

while not done:
    action = env.action_space.sample()  # Random action
    obs, reward, done, info = env.step(action)
    env.render()
