
#  Stock Market Simulator (PPO Agent)

A Deep Reinforcement Learning-based Stock Market Simulator using a PPO (Proximal Policy Optimization) agent to simulate trading decisions on real stock market data.

##  Screenshots

![App Screenshot 1](image.png)
![App Screenshot 2](3be9b15d01ac7c0e2f75d0713801c85cbcca9364337afc81f9d5bac7.png)

---

##  Features

- Upload stock CSV data
- Preprocess and clean data
- Simulate trades using a trained PPO agent
- Visualize:
  - Portfolio value over time 📈
  - Buy/Sell action points 📍
- Fully interactive Streamlit web app interface

---

##  Project Structure

```
stock-rl/
├── env/
│   └── trading_env.py        # Custom Gym-like trading environment
├── model/
│   └── ppo_tatasteel.zip     # Trained PPO model
├── data/
│   └── tatasteel_daily.csv   # Sample stock CSV data
├── train_ppo.py              # PPO model training script
├── streamlit_app.py          # Streamlit frontend app
└── README.md
```

---

## 🧠 Model Training

The PPO model was trained using [Stable-Baselines3](https://github.com/DLR-RM/stable-baselines3) with a custom trading environment.

### Hyperparameters:

- Policy: `MlpPolicy`
- Total Timesteps: `100_000`
- Learning Rate: `3e-4`
- Gamma: `0.99`

---

##  Run Locally

### ⚙️ Setup Environment

```bash
# Create virtual environment
python3 -m venv stocks-rl
source stocks-rl/bin/activate

# Install dependencies
pip install -r requirements.txt
```

###  Run Streamlit App

```bash
streamlit run streamlit_app.py
```

> Make sure to place your trained model in the `model/` directory and stock CSV file in the `data/` folder.

---

##  Tech Used

- Python 
- Streamlit 
- Stable-Baselines3 
- Gymnasium 
- Matplotlib & Seaborn 
- Pandas, NumPy 🔢

---

## 📜 License

[MIT License](LICENSE)
