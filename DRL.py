import numpy as np
import pandas as pd
import tensorflow as tf
from stable_baselines3 import PPO
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import gymnasium as gym
from gymnasium import spaces

# Load and preprocess data
try:
    df = pd.read_csv("ICICI_2019_to_2024_all_sentiment.csv", parse_dates=['Date'], dayfirst=True)
    print("Data loaded successfully. First 5 rows:")
    print(df.head())
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# Verify critical columns exist
required_columns = {
    'Date', 'Open Price', 'High Price', 'Low Price', 'Close Price', 'Volume',
    'ET_net_sentiment_score', 'Indian_sentiment_score', 'foreign_sentiment_score'
}

if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    print(f"Missing columns: {missing}")
    exit()

# Feature Engineering
def add_technical_indicators(df):
    """Adds technical indicators and sentiment-based transformations"""
    # MACD
    ema_12 = df['Close Price'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close Price'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26

    # Sentiment Moving Averages
    df['Indian_Sentiment_SMA_14'] = df['Indian_sentiment_score'].rolling(window=14).mean()
    df['ET_Sentiment_SMA_14'] = df['ET_net_sentiment_score'].rolling(window=14).mean()
    df['Foreign_Sentiment_SMA_14'] = df['foreign_sentiment_score'].rolling(window=14).mean()

    # Bollinger Bands
    df['BB_Mid'] = df['Close Price'].rolling(window=20).mean()
    df['BB_Upper'] = df['BB_Mid'] + (df['Close Price'].rolling(window=20).std() * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['Close Price'].rolling(window=20).std() * 2)
    df['BB_Width'] = df['BB_Upper'] - df['BB_Lower']

    # Lag Features
    df['Open_Lag_1'] = df['Open Price'].shift(1)
    df['Log_Volume'] = np.log1p(df['Volume'])

    # Stochastic Oscillator
    low_min = df['Low Price'].rolling(window=14).min()
    high_max = df['High Price'].rolling(window=14).max()
    df['%K'] = ((df['Close Price'] - low_min) / (high_max - low_min)) * 100

    # Williams %R (WPR)
    df['WPR'] = ((high_max - df['Close Price']) / (high_max - low_min)) * -100
    df['OBV'] = (np.sign(df['Close Price'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Volume Weighted Average Price (VWAP)
    df['VWAP'] = (df['Close Price'] * df['Volume']).cumsum() / df['Volume'].cumsum()

    return df.dropna()

df = add_technical_indicators(df)

def create_rolling_sequences(data, window_size):
    """Returns proper numpy arrays for LSTM training"""
    X, y = [], []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
        y.append(data[i + window_size, 0])  # Predicting 'Open Price'
    
    X = np.array(X)
    y = np.array(y)
    
    if len(X.shape) == 2:
        X = np.expand_dims(X, axis=-1)
        
    return X, y

def train_lstm(df, date_d):
    if date_d not in df['Date'].astype(str).values:
        raise ValueError(f"Error: No data found for the selected test date {date_d}")

    date_d = pd.to_datetime(date_d)
    train_df = df[df['Date'] < date_d].copy()
    
    min_days_required = 10
    if len(train_df) < min_days_required:
        print(f"\nERROR: Need at least {min_days_required} days for training.")
        print(f"Available days before {date_d.date()}: {len(train_df)}")
        print(f"Earliest possible test date: {df['Date'].iloc[min_days_required].date()}")
        return None, None, None

    feature_columns = [
        'Open Price', 'Close Price', 'Log_Volume',
        'Open_Lag_1', 'MACD', 'BB_Width',
        'Indian_Sentiment_SMA_14', 'Foreign_Sentiment_SMA_14'
    ]
    
    train_features = train_df[feature_columns].values
    
    if np.isnan(train_features).any():
        train_features = np.nan_to_num(train_features)
    
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_features)
    
    X_train, y_train = create_rolling_sequences(train_scaled, window_size=10)
    
    print(f"\nTraining data prepared:")
    print(f"- Total days available: {len(train_df)}")
    print(f"- Training sequences: {len(X_train)}")
    print(f"- Input shape: {X_train.shape}")
    
    if len(X_train) > 0:
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
            LSTM(64),
            Dense(1)
        ])
        model.compile(optimizer='adam', loss='mse')
        history = model.fit(X_train, y_train, epochs=20, batch_size=32, verbose=0)
        pbar = tqdm(range(20))
        for epoch in pbar:
            loss = history.history['loss'][epoch]
            pbar.set_description(f"Epoch {epoch+1} | Loss: {loss:.4f}")
        return model, scaler, train_scaled
    return None, None, None

def predict_next_day(model, scaler, df, date_d):
    if model is None or scaler is None:
        raise ValueError("Model/scaler not initialized")
    
    try:
        date_d = pd.to_datetime(date_d)
        test_df = df[df['Date'] == date_d].copy()
        
        if len(test_df) == 0:
            raise ValueError(f"No data available for {date_d.date()}")
            
        feature_columns = [
            'Open Price', 'Close Price', 'Log_Volume',
            'Open_Lag_1', 'MACD', 'BB_Width',
            'Indian_Sentiment_SMA_14', 'Foreign_Sentiment_SMA_14'
        ]
        
        test_features = test_df[feature_columns].values
        
        # Get the last 10 days of data including test date
        date_idx = df[df['Date'] == date_d].index[0]
        if date_idx < 10:
            raise ValueError("Not enough historical data for prediction")
        
        historical_data = df.iloc[date_idx-10:date_idx][feature_columns].values
        historical_scaled = scaler.transform(historical_data)
        
        # Reshape for LSTM (1 sample, 10 timesteps, n_features)
        X_test = historical_scaled.reshape(1, 10, historical_scaled.shape[1])
        
        y_pred = model.predict(X_test)
        
        # Inverse transform
        dummy = np.zeros((1, historical_scaled.shape[1]))
        dummy[:, 0] = y_pred.flatten()
        predicted_price = scaler.inverse_transform(dummy)[0, 0]
        
        return predicted_price
            
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return None
class StockTradingEnv(gym.Env):
    def _init_(self, df, data_scaled):
        super()._init_()
        self.df = df.reset_index()
        self.data_scaled = data_scaled
        self.current_step = 10
        self.shares_held = 0
        self.initial_cash = 100000
        self.cash = self.initial_cash
        
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(10, data_scaled.shape[1]),
            dtype=np.float32
        )
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 10
        self.shares_held = 0
        self.cash = self.initial_cash
        observation = self._get_obs()
        info = {}  
        return observation, info
    
    def _get_obs(self):
        return self.data_scaled[self.current_step-10:self.current_step]
    
    def step(self, action):
        if self.current_step >= len(self.df):
            print(f"Warning: current_step {self.current_step} exceeds DataFrame length {len(self.df)}")
            return self._get_obs(), 0, True, False, {}

        current_price = self.df.iloc[self.current_step]['Open Price']
    
        # Execute trade
        shares_traded = int(action[0] * 10)
        cost = shares_traded * current_price
        
        if shares_traded > 0 and self.cash >= cost:  # Buy
            self.shares_held += shares_traded
            self.cash -= cost
        elif shares_traded < 0 and self.shares_held >= abs(shares_traded):  # Sell
            self.shares_held += shares_traded
            self.cash += abs(cost)  # Corrected from subtracting cost
        
        self.current_step += 1
        terminated = self.current_step >= len(self.df) - 1
        truncated = False  
        
        next_price = self.df.iloc[self.current_step]['Open Price']
        portfolio_value = self.shares_held * next_price + self.cash
        reward = (portfolio_value - self.initial_cash) / self.initial_cash  # Normalized reward
        
        observation = self._get_obs()
        info = {
            'shares_held': self.shares_held,
            'cash': self.cash,
            'portfolio_value': portfolio_value
        }
        
        return observation, reward, terminated, truncated, info

def train_drl_agent(df, model, scaler, date_d, shares_held, data_scaled):
    test_df = df[df['Date'] == pd.to_datetime(date_d)].copy()
    if len(test_df) == 0:
        raise ValueError(f"No data available for {date_d}")
    
    env = StockTradingEnv(test_df, data_scaled)
    
    # Check if shimmy is installed
    try:
        import shimmy
    except ImportError:
        raise ImportError("Please install shimmy: pip install 'shimmy>=2.0'")
    
    agent = PPO('MlpPolicy', env, verbose=1)
    agent.learn(total_timesteps=2000)
    
    obs, _ = env.reset()
    action, _ = agent.predict(obs)
    
    current_price = test_df.iloc[0]['Open Price']
    predicted_price = predict_next_day(model, scaler, df, date_d)
    
    action_value = action[0]  # Fix for array indexing
    
    if action_value > 0.33:
        signal = "BUY"
        shares_traded = +int(shares_held * action_value)
    elif action_value < -0.33:
        signal = "SELL"
        shares_traded = -int(shares_held * abs(action_value))
    else:
        signal = "HOLD"
        shares_traded = 0
    
    net_worth = shares_held * current_price
    profit = shares_held * (predicted_price - current_price) if predicted_price else 0
    
    return signal, net_worth, profit, shares_traded

# Main execution
print(f"\nAvailable date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Minimum training requirement: 10 days")

while True:
    try:
        date_d = input(f"Enter test date (YYYY-MM-DD) between {df['Date'].iloc[10].date()} and {df['Date'].iloc[-1].date()}: ")
        test_date = pd.to_datetime(date_d)
        if test_date < df['Date'].iloc[10] or test_date > df['Date'].iloc[-1]:
            print(f"Date must have at least 10 prior trading days")
            continue
        break
    except ValueError:
        print("Invalid date format")

shares_held = float(input(f"Enter shares held on {date_d}: "))

try:
    # Train LSTM
    model, scaler, data_scaled = train_lstm(df, date_d)
    if model is None:
        exit()
    
    # Get prediction
    predicted_price = predict_next_day(model, scaler, df, date_d)
    if predicted_price is None:
        exit()
    
    print(f"\nPredicted next open price: {predicted_price:.2f}")
    
    # Train DRL agent and get recommendation
    signal, net_worth, profit, shares_traded = train_drl_agent(
        df, model, scaler, date_d, shares_held, data_scaled)
    
    current_price = df[df['Date'] == pd.to_datetime(date_d)].iloc[0]['Open Price']
    
    print("\n=== Trading Recommendation ===")
    print(f"Current Price: {current_price:.2f}")
    print(f"Predicted Price: {predicted_price:.2f}")
    print(f"Action: {signal}")
    print(f"Shares to Trade: {shares_traded:+}")
    print(f"Current Holdings: {shares_held}")
    print(f"Current Net Worth: ₹{net_worth:,.2f}")
    print(f"Expected Profit: ₹{profit:,.2f}")

except Exception as e:
    print(f"\nError during trading simulation: {str(e)}")