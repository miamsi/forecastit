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
st.markdown("Version 1: Ticker History Fix (Bypasses Streamlit IP Blocking)")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Machine Settings")
ticker = st.sidebar.text_input("Stock Ticker (use .JK for IHSG)", value="BBRI.JK")
lookback_years = st.sidebar.slider("Training History (Years)", 1, 5, 2)

# --- 1. DATA INGESTION ENGINE (Version 1 - Ticker History Fix) ---
@st.cache_data(ttl=3600)
def fetch_data(symbol, years):
    try:
        # FIX: yf.download() silently fails on Streamlit Cloud. 
        # yf.Ticker().history() is much more robust and never returns Multi-Index!
        stock = yf.Ticker(symbol)
        data = stock.history(period=f"{years}y", interval="1d")
        
        if data.empty:
            return None
            
        # Ensure we have the required columns
        df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
        
        # Calculate features
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        df['MA_20'] = df['Close'].rolling(window=20).mean()
        df = df.dropna()
        return df
    except Exception as e:
        st.sidebar.error(f"Data Fetch Error: {e}")
        return None

# --- 2. HMM REGIME DETECTION ---
@st.cache_data
def detect_regimes(log_returns):
    X = log_returns.values.reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=2, covariance_type="full", n_iter=1000, random_state=42)
    model.fit(X)
    hidden_states = model.predict(X)
    
    var0 = np.var(X[hidden_states == 0])
    var1 = np.var(X[hidden_states == 1])
    volatile_state = 0 if var0 > var1 else 1
    
    regime_labels = ["Volatile" if s == volatile_state else "Trending" for s in hidden_states]
    return regime_labels, hidden_states == volatile_state

# --- 3. GARCH VOLATILITY FORECAST ---
@st.cache_data
def forecast_volatility(returns):
    scaled_returns = returns * 100
    am = arch_model(scaled_returns, vol='Garch', p=1, q=1, dist='normal')
    res = am.fit(disp='off')
    forecasts = res.forecast(horizon=1)
    # Correctly access the variance from the result object
    next_day_var = forecasts.variance.values[-1, 0]
    return np.sqrt(next_day_var) / 100

# --- MAIN EXECUTION ---
df = fetch_data(ticker, lookback_years)

if df is not None:
    with st.spinner("Calculating Volatility Regimes..."):
        df['Regime'], df['Is_Volatile'] = detect_regimes(df['Log_Returns'])
        next_day_risk = forecast_volatility(df['Returns'])
        
    # --- DASHBOARD UI ---
    current_price = float(df['Close'].iloc[-1])
    current_regime = df['Regime'].iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"Rp {current_price:,.0f}")
    
    regime_status = "🔴 Panic/High Risk" if df['Is_Volatile'].iloc[-1] else "🟢 Calm/Trending"
    col2.metric("Market Regime", regime_status)
    
    col3.metric("Tomorrow's Risk", f"± {next_day_risk*100:.2f}%")

    # --- PLOTTING ---
    fig = go.Figure()
    
    # Candlestick
    fig.add_trace(go.Candlestick(x=df.index, open=df['Open'], high=df['High'], 
                                 low=df['Low'], close=df['Close'], name='Price'))
    
    # Moving Average
    fig.add_trace(go.Scatter(x=df.index, y=df['MA_20'], name='20-Day MA', 
                             line=dict(color='cyan', width=1.5)))

    # Red background for volatile periods
    volatile_periods = df[df['Is_Volatile']].index
    for date in volatile_periods:
        fig.add_vrect(x0=date, x1=date + pd.Timedelta(days=1), 
                      fillcolor="red", opacity=0.05, line_width=0)

    fig.update_layout(template="plotly_dark", height=600, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)

    # The Machine's Logic
    if df['Is_Volatile'].iloc[-1]:
        st.warning(f"⚠️ **Regime Alert:** {ticker} is in a high-volatility state. Moving Averages are likely to be 'whiplashed.' Tighten stops to {next_day_risk*100:.2f}%.")
    else:
        st.info("✅ **Market Calm:** Trend following with Moving Averages is currently high-probability.")

else:
    st.error("Could not load data. Please refresh or check the ticker format.")
