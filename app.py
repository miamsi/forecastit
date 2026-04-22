import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from hmmlearn import hmm
from arch import arch_model
import plotly.graph_objects as go

# --- PAGE CONFIG ---
st.set_page_config(page_title="IHSG Volatility Machine", layout="wide")
st.title("🤖 IHSG Volatility-Aware Trend Machine")
st.markdown("Combines Moving Averages with **Hidden Markov Models (HMM)** and **GARCH** to detect market panic before trends break.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Machine Settings")
ticker = st.sidebar.text_input("Stock Ticker (use .JK for IHSG)", value="BBCA.JK")
lookback_years = st.sidebar.slider("Training History (Years)", 1, 5, 2)

# --- 1. DATA INGESTION ENGINE (Cached) ---
@st.cache_data(ttl=3600) # Cache for 1 hour
def fetch_data(symbol, years):
    df = yf.download(symbol, period=f"{years}y", interval="1d")
    if df.empty:
        return None
    
    # Calculate essential features
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_60'] = df['Close'].rolling(window=60).mean()
    df = df.dropna()
    return df

# --- 2. HMM REGIME DETECTION ENGINE (Cached) ---
@st.cache_data
def detect_regimes(log_returns):
    # Reshape for HMM
    X = log_returns.values.reshape(-1, 1)
    
    # Train a 2-state Hidden Markov Model
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    
    # HMM doesn't know which state is "Panic". We figure it out by checking variance.
    var_state_0 = np.var(X[hidden_states == 0])
    var_state_1 = np.var(X[hidden_states == 1])
    
    volatile_state = 0 if var_state_0 > var_state_1 else 1
    
    # Create readable labels
    regime_labels = ["Volatile/Panic" if s == volatile_state else "Calm/Trending" for s in hidden_states]
    return regime_labels, hidden_states == volatile_state

# --- 3. GARCH VOLATILITY FORECAST ENGINE (Cached) ---
@st.cache_data
def forecast_volatility(returns):
    # GARCH needs scaled returns (x100) to converge properly
    scaled_returns = returns * 100
    am = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    
    # Forecast next day variance
    forecasts = res.forecast(horizon=1)
    next_day_vol = np.sqrt(forecasts.variance.values[-1, :][0]) / 100 # scale back
    return next_day_vol

# --- MAIN EXECUTION ---
df = fetch_data(ticker, lookback_years)

if df is not None:
    # Run the advanced math
    with st.spinner("Training HMM and GARCH models..."):
        df['Regime'], df['Is_Volatile'] = detect_regimes(df['Log_Returns'])
        next_day_risk = forecast_volatility(df['Returns'])
        
    # --- DASHBOARD UI ---
    current_price = df['Close'].iloc[-1]
    current_regime = df['Regime'].iloc[-1]
    current_ma20 = df['MA_20'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"Rp {current_price:,.0f}")
    
    # Dynamic styling for regime
    regime_color = "🔴" if df['Is_Volatile'].iloc[-1] else "🟢"
    col2.metric("Current Market Regime (HMM)", f"{regime_color} {current_regime}")
    
    # Show GARCH forecast as a percentage movement risk
    col3.metric("Next Day Risk (GARCH Volatility)", f"± {next_day_risk*100:.2f}%")

    # --- MACHINE LOGIC / CIRCUIT BREAKER ---
    st.markdown("### 🧠 The Machine's Verdict")
    if df['Is_Volatile'].iloc[-1]:
        st.error(f"**WARNING:** The HMM detects a High Volatility Regime. Moving Averages (like the 20-day MA at Rp {current_ma20:,.0f}) are currently UNRELIABLE. Use GARCH risk parameters (± {next_day_risk*100:.2f}%) for stop-losses.")
    else:
        st.success(f"**ALL CLEAR:** The market is Calm. Standard technical analysis and Moving Averages are highly reliable right now. You can trust the trend.")

    # --- PLOTTING ---
    st.markdown("### 📈 Interactive Regime Chart")
    fig = go.Figure()

    # Candlestick chart
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'],
                    name='Price'))

    # Moving Average
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], 
                             line=dict(color='blue', width=1.5), 
                             name='20-Day MA'))

    # Highlight Volatile Regimes with background colors
    volatile_dates = df[df['Is_Volatile']].index
    for date in volatile_dates:
        fig.add_vrect(x0=date, x1=date + pd.Timedelta(days=1), 
                      fillcolor="red", opacity=0.1, line_width=0)

    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

else:
    st.error("Failed to fetch data. Please check the ticker symbol (ensure you use .JK for Indonesian stocks).")
