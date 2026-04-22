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
st.markdown("Version 1: Handles Multi-Index Data & Yahoo Finance rate-limiting.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Machine Settings")
ticker = st.sidebar.text_input("Stock Ticker (use .JK for IHSG)", value="PTBA.JK")
lookback_years = st.sidebar.slider("Training History (Years)", 1, 5, 2)

# --- 1. DATA INGESTION ENGINE (Updated for 2026 Multi-Index) ---
@st.cache_data(ttl=3600)
def fetch_data(symbol, years):
    try:
        # Fetch data with multi-index fix
        data = yf.download(symbol, period=f"{years}y", interval="1d", multi_level_index=False)
        
        if data.empty:
            return None
            
        df = data.copy()
        
        # Calculate essential features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df['MA_60'] = df['Close'].rolling(window=60).mean()
        df = df.dropna()
        return df
    except Exception as e:
        st.sidebar.error(f"Error: {e}")
        return None

# --- 2. HMM REGIME DETECTION ENGINE ---
@st.cache_data
def detect_regimes(log_returns):
    X = log_returns.values.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    
    var_state_0 = np.var(X[hidden_states == 0])
    var_state_1 = np.var(X[hidden_states == 1])
    volatile_state = 0 if var_state_0 > var_state_1 else 1
    
    regime_labels = ["Volatile/Panic" if s == volatile_state else "Calm/Trending" for s in hidden_states]
    return regime_labels, hidden_states == volatile_state

# --- 3. GARCH VOLATILITY FORECAST ENGINE ---
@st.cache_data
def forecast_volatility(returns):
    scaled_returns = returns * 100
    am = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    forecasts = res.forecast(horizon=1)
    next_day_vol = np.sqrt(forecasts.variance.values[-1, :][0]) / 100
    return next_day_vol

# --- MAIN EXECUTION ---
df = fetch_data(ticker, lookback_years)

if df is not None:
    with st.spinner("Analyzing Market States..."):
        df['Regime'], df['Is_Volatile'] = detect_regimes(df['Log_Returns'])
        next_day_risk = forecast_volatility(df['Returns'])
        
    # --- DASHBOARD UI ---
    current_price = df['Close'].iloc[-1]
    current_regime = df['Regime'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"Rp {current_price:,.0f}")
    regime_color = "🔴" if df['Is_Volatile'].iloc[-1] else "🟢"
    col2.metric("Regime", f"{regime_color} {current_regime}")
    col3.metric("Next Day Risk", f"± {next_day_risk*100:.2f}%")

    # --- CHART ---
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Price', line=dict(color='white')))
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], name='20 MA', line=dict(color='cyan', dash='dot')))
    
    # Shade volatile areas
    volatile_df = df[df['Is_Volatile']]
    if not volatile_df.empty:
        for i in range(len(volatile_df)):
            fig.add_vrect(x0=volatile_df.index[i], x1=volatile_df.index[i] + pd.Timedelta(days=1), 
                          fillcolor="red", opacity=0.1, line_width=0)

    fig.update_layout(template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.error("Still unable to fetch PTBA.JK. This usually means Yahoo is blocking the Streamlit IP. Try running it locally first or wait a few minutes.")
