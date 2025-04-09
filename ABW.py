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
    def __init__(self, df, data_scaled):
        super().__init__()
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

def train_drl_agent_historical(df, data_scaled, date_d):
    """Trains the DRL agent on all data before the selected date."""
    date_d_dt = pd.to_datetime(date_d)
    train_df_rl = df[df['Date'] < date_d_dt].copy().reset_index(drop=True)

    if len(train_df_rl) < 11:  # Minimum for initial observation + 1 step
        raise ValueError(f"Insufficient historical data ({len(train_df_rl)} days) before {date_d} for DRL training.")

    # Use the entire scaled data before date_d for the environment
    train_scaled_rl = data_scaled[:len(train_df_rl)]

    env_train_rl = StockTradingEnv(train_df_rl, train_scaled_rl)

    try:
        import shimmy
    except ImportError:
        raise ImportError("Please install shimmy: pip install 'shimmy>=2.0'")

    agent = PPO('MlpPolicy', env_train_rl, verbose=1)
    agent.learn(total_timesteps=10000)  # Train for more timesteps on historical data

    return agent

def test_drl_agent_on_day_after(df, agent, scaler, date_d, shares_held):
    """Tests the trained DRL agent on the day after the selected date."""
    date_d_dt = pd.to_datetime(date_d)
    day_after = date_d_dt + pd.Timedelta(days=1)
    test_df_next_day = df[df['Date'] == day_after].copy().reset_index(drop=True)
    test_df_current_day = df[df['Date'] == date_d].copy().reset_index(drop=True)

    if len(test_df_next_day) == 0:
        print(f"No data available for the day after {date_d.date()}")
        return None, None, None, None

    if len(test_df_current_day) == 0:
        print(f"No data available for {date_d.date()}")
        return None, None, None, None

    # Prepare observation for the agent using historical data leading up to the test day
    history_len = 10
    test_date_index_in_original_df = df[df['Date'] == day_after].index[0]

    if test_date_index_in_original_df < history_len:
        print(f"Insufficient history before {day_after.date()} for observation.")
        return None, None, None, None

    historical_data = df.iloc[test_date_index_in_original_df - history_len : test_date_index_in_original_df][['Open Price', 'Close Price', 'Log_Volume', 'Open_Lag_1', 'MACD', 'BB_Width', 'Indian_Sentiment_SMA_14', 'Foreign_Sentiment_SMA_14']].values
    historical_scaled = scaler.transform(historical_data)

    # Create a dummy environment for getting the action
    dummy_env = StockTradingEnv(test_df_next_day, np.zeros((1, historical_scaled.shape[1]))) # df has 1 row
    obs, _ = dummy_env.reset() # Reset to get initial state (not used for prediction directly here)

    # Predict action based on the historical scaled data
    action, _ = agent.predict(historical_scaled)

    current_price_date_d_close = test_df_current_day.iloc[0]['Close Price']
    predicted_price_next_day_open = predict_next_day(model, scaler, df, day_after.strftime('%Y-%m-%d'))

    action_value = action[0]

    if action_value > 0.33:
        signal = "BUY"
        shares_traded = +int(shares_held * action_value)
    elif action_value < -0.33:
        signal = "SELL"
        shares_traded = -int(shares_held * abs(action_value))
    else:
        signal = "HOLD"
        shares_traded = 0

    net_worth_date_d_close = shares_held * current_price_date_d_close
    expected_profit_next_day = net_worth_date_d_close + (shares_traded * predicted_price_next_day_open)

    return signal, net_worth_date_d_close, expected_profit_next_day, shares_traded

# Main execution (partial - only the relevant part with changes)
# Main execution (partial - only the relevant part with changes)
print(f"\nAvailable date range: {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Minimum training requirement: 10 days")

while True:
    try:
        date_d = input(f"Enter test date (YYYY-MM-DD) between {df['Date'].iloc[10].date()} and {df['Date'].iloc[-2].date()}: ") # Adjusted range for testing on the next day
        test_date = pd.to_datetime(date_d)
        if test_date < df['Date'].iloc[10] or test_date >= df['Date'].iloc[-1]: # Adjusted condition
            print(f"Date must have at least 10 prior trading days and a following day for testing")
            continue
        break
    except ValueError:
        print("Invalid date format")

shares_held = float(input(f"Enter shares held on {date_d}: "))

try:
    # Train LSTM (no changes here)
    model, scaler, data_scaled = train_lstm(df, date_d)
    if model is None:
        exit()

    # Train DRL agent on historical data (no changes here)
    drl_agent = train_drl_agent_historical(df, data_scaled, date_d)
    if drl_agent is None:
        exit()

    # Test DRL agent on the day after the selected date (function with changes)
    signal, net_worth_date_d_close, expected_profit_next_day, shares_traded = test_drl_agent_on_day_after(
        df, drl_agent, scaler, date_d, shares_held
    )

    if signal:
        day_after_date = pd.to_datetime(date_d) + pd.Timedelta(days=1)
        actual_open_price_next_day = df[df['Date'] == day_after_date].iloc[0]['Open Price'] if len(df[df['Date'] == day_after_date]) > 0 else None
        predicted_price_next_day_open = predict_next_day(model, scaler, df, day_after_date.strftime('%Y-%m-%d')) # Use day_after_date here

        print("\n=== DRL Agent Trading Recommendation (Day After) ===")
        if actual_open_price_next_day is not None:
            print(f"Actual Open Price (Day After): {actual_open_price_next_day:.2f}")
        if predicted_price_next_day_open is not None:
            print(f"Predicted Open Price (Day After): {predicted_price_next_day_open:.2f}")
        print(f"Action: {signal}")
        print(f"Shares to Trade: {shares_traded:+}")
        print(f"Current Holdings (on {pd.to_datetime(date_d).date()}): {shares_held}")
        print(f"Portfolio Value on {pd.to_datetime(date_d).date()} Closing: ₹{net_worth_date_d_close:,.2f}")
        print(f"Expected PnL on Next Day Open: ₹{expected_profit_next_day:,.2f}")
        
except Exception as e:
    print(f"\nError during training or testing: {str(e)}")
