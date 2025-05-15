import streamlit as st
import pandas as pd
import time
from SmartApi import SmartConnect
import pyotp
from datetime import datetime, timedelta
from SmartApi.smartExceptions import DataException

# ---------------- Streamlit Configuration ----------------
st.set_page_config(page_title="Intraday Breakout & Options Dashboard", layout="wide")

# ------------- Load Credentials -------------
# Create `.streamlit/secrets.toml` with your SmartAPI credentials:
# SMARTAPI_KEY = "..."
# CLIENT_CODE   = "..."
# PASSWORD      = "..."
# TOTP_SEED     = "..."
SMARTAPI_KEY = st.secrets.get("SMARTAPI_KEY")
CLIENT_CODE  = st.secrets.get("CLIENT_CODE")
PASSWORD     = st.secrets.get("PASSWORD")
TOTP_SEED    = st.secrets.get("TOTP_SEED")

if not all([SMARTAPI_KEY, CLIENT_CODE, PASSWORD, TOTP_SEED]):
    st.error("Please set all SmartAPI credentials in .streamlit/secrets.toml")
    st.stop()

# ----------- Initialize & Authenticate SmartAPI -----------
@st.cache_resource
def init_smartapi():
    client = SmartConnect(api_key=SMARTAPI_KEY)
    totp_code = pyotp.TOTP(TOTP_SEED).now()
    client.generateSession(CLIENT_CODE, PASSWORD, totp_code)
    return client

smart_api = init_smartapi()

# ------------ Helper: Rate-limited API call -------------
def call_with_retry(func, *args, max_retries=5, delay=1.5, **kwargs):
    """Retry SmartAPI calls on rate-limit errors with exponential backoff."""
    for _ in range(max_retries):
        try:
            return func(*args, **kwargs)
        except DataException as e:
            if 'exceeding access rate' in str(e):
                time.sleep(delay)
                delay *= 2
            else:
                raise
    st.error("Max retries exceeded due to rate limiting.")
    return None

@st.cache_data(show_spinner=False)
def fetch_candle_data(token: str, interval: str, hours: int = 5) -> pd.DataFrame:
    to_dt = datetime.now()
    from_dt = to_dt - timedelta(hours=hours)

    interval_map = {
        '1m': 'ONE_MINUTE',
        '5m': 'FIVE_MINUTE',
        '15m': 'FIFTEEN_MINUTE',
        '30m': 'THIRTY_MINUTE'
    }

    params = {
        "exchange": "NSE",
        "symboltoken": token,
        "interval": interval_map[interval],
        "fromdate": from_dt.strftime("%Y-%m-%d %H:%M"),  # ✅ Format like "2021-02-08 09:00"
        "todate": to_dt.strftime("%Y-%m-%d %H:%M")
    }

    print("Fetching with params:", params)

    resp = call_with_retry(smart_api.getCandleData, params)
    data = resp.get('data') if resp else []

    if not data:
        return pd.DataFrame()  # Return empty DataFrame if no data

    df = pd.DataFrame(data, columns=["ts", "O", "H", "L", "C", "V"])
    df['ts'] = pd.to_datetime(df['ts'], format="%Y-%m-%d %H:%M:%S", errors='coerce')  # Coerce invalids to NaT
    df.dropna(subset=["ts"], inplace=True)  # Drop bad rows if timestamp parsing fails
    df.set_index('ts', inplace=True)
    df.rename(columns={'O': 'open', 'H': 'high', 'L': 'low', 'C': 'close', 'V': 'volume'}, inplace=True)
    return df

# ------- Utility: Fetch Option Chain -------
@st.cache_data(show_spinner=False)
def fetch_option_chain(symbol: str, expiry: str) -> pd.DataFrame:
    """
    Fetch intraday option chain for given symbol & expiry (YYYY-MM-DD).
    """
    resp = call_with_retry(smart_api.getOptionChain, "NFO", symbol, expiry)
    data = resp.get('data', []) if resp else []
    return pd.json_normalize(data)

# ------- Breakout Detection -------
def detect_breakout(df: pd.DataFrame, mult: float) -> dict:
    """
    Returns breakout info if last candle closes above prior day’s high on volume spike.
    """
    intraday = df.between_time('09:15','15:30')
    prev_day = intraday.last('1D').iloc[:-1]
    if prev_day.empty:
        return {}
    prev_high = prev_day['high'].max()
    vol_avg   = df['volume'].rolling(30).mean().iloc[-2]
    last      = df.iloc[-1]
    if last['close'] > prev_high and last['volume'] >= mult * vol_avg:
        return {'high': prev_high, 'close': last['close'], 'vol': last['volume'], 'avg': vol_avg}
    return {}

# ---------------- Streamlit UI ----------------
st.title("📈 Intraday Breakout & Options Dashboard")

with st.sidebar:
    symbols_input = st.text_input(
        "Symbols (comma-separated)",
        "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,KOTAKBANK,SBIN,AXISBANK,ITC,LT,MARUTI,BAJFINANCE,TECHM"
    )
    symbols = [s.strip().upper() for s in symbols_input.split(',') if s.strip()]
    interval = st.selectbox("Interval", ['1m','5m','15m','30m'], index=1)
    vol_mult = st.slider("Volume Multiplier", 1.0, 5.0, 2.0, step=0.5)
    expiry   = st.date_input("Option Expiry", datetime.now().date())
    run_scan = st.button("🔍 Scan Breakouts")

if run_scan:
    results = []
    for sym in symbols:
        # 1) Lookup symbol token
        sdata = call_with_retry(smart_api.searchScrip, "NSE", sym)
        token = sdata.get('data',[{}])[0].get('symboltoken') if sdata else None
        if not token:
            st.warning(f"Token not found for {sym}")
            continue

        # 2) Fetch candles & detect breakout
        df   = fetch_candle_data(token, interval)
        info = detect_breakout(df, vol_mult)
        if not info:
            continue

        # 3) Fetch option chain & compute ATM prices
        opt_df    = fetch_option_chain(sym, expiry.strftime('%Y-%m-%d'))
        strikes   = opt_df['strikePrice'].unique().tolist()
        atm_strike= min(strikes, key=lambda x: abs(x - info['close']))
        ce_price  = opt_df.loc[opt_df.strikePrice==atm_strike, 'CE.lastPrice'].iloc[0]
        pe_price  = opt_df.loc[opt_df.strikePrice==atm_strike, 'PE.lastPrice'].iloc[0]

        results.append({
            'Symbol':      sym,
            'Prev High':   info['high'],
            'Close':       info['close'],
            'Volume':      int(info['vol']),
            'Vol Avg':     int(info['avg']),
            'ATM Strike':  atm_strike,
            'CE Price':    ce_price,
            'PE Price':    pe_price
        })

    if results:
        df_res = pd.DataFrame(results)
        st.success(f"Detected breakouts in {len(results)} symbol(s)")
        st.dataframe(df_res)

        # Plot first symbol’s price chart
        first = results[0]['Symbol']
        sdata = call_with_retry(smart_api.searchScrip, "NSE", first)
        token = sdata.get('data',[{}])[0].get('symboltoken')
        chart_df = fetch_candle_data(token, interval)
        st.subheader(f"Price Chart: {first}")
        st.line_chart(chart_df['close'])
    else:
        st.warning("No breakouts detected with current settings.")

st.markdown("---")
st.markdown("*This app uses epoch-ms formatting for `getCandleData` and exponential backoff for rate-limit resilience.*")
