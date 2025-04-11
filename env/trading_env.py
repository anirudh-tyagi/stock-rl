import gym
from gym import spaces
import numpy as np
import pandas as pd

class TradingEnv(gym.Env):

    def _take_action(self, action):
        current_price = self.df.loc[self.current_step, 'Price']
        if action == 1:  # Buy
            if self.cash >= current_price:
                self.stock_held += 1
                self.cash -= current_price
            elif action == 2:  # Sell
                if self.stock_held > 0:
                    self.stock_held -= 1
                    self.cash += current_price
    # action == 0 is Hold — do nothing

    def __init__(self, df, initial_cash=100000):
        super(TradingEnv, self).__init__()
        self.df = df.reset_index()
        self.initial_cash = initial_cash

        # Action space: 0 = Hold, 1 = Buy, 2 = Sell
        self.action_space = spaces.Discrete(3)

        # Observation space: [Open, High, Low, Close, Volume, MA10, MA50, Cash, Stock Held]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )

        self.reset()

    def reset(self):
        self.current_step = 0
        self.stock_held = 0
        self.cash = 100000
        self.total_value = self.cash
        self.prev_total_value = self.cash 
        return self._get_obs()

    def _get_obs(self):
        row = self.df.iloc[self.current_step]
        obs = np.array([
            row['Open'],
            row['High'],
            row['Low'],
            row['Close'],
            row['Volume'],
            row['MA10'],
            row['MA50'],
            self.cash,
            self.stock_held
        ], dtype=np.float32)
        return obs

    def step(self, action):
        self._take_action(action)
        self.current_step += 1
        if self.current_step > len(self.df) - 1:
            self.current_step = len(self.df) - 1
            done = True
        else:
            done = False
        obs = self._get_obs()
        # Get current price
        price = self.df.loc[self.current_step, 'Price']
        # Calculate total value and reward
        total_value = self.cash + self.stock_held * price
        reward = total_value - self.prev_total_value
        self.prev_total_value = total_value
        info = {
        'cash': self.cash,
        'stock_held': self.stock_held,
        'value': total_value,
        'price': price  # ✅ This is the required fix
        }
        return obs, reward, done, info

    
    def render(self, mode='human'):
        print(f"Step: {self.current_step}")
        print(f"Price: {self.df.iloc[self.current_step]['Close']:.2f}")
        print(f"Cash: {self.cash:.2f}, Stock Held: {self.stock_held}, Total Value: {self.total_value:.2f}")
