from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import warnings
import requests
import pytz
from datetime import datetime

# --- CONFIGURATION ---
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

app = Flask(__name__)

# --- FIREBASE SETUP (The Cloud Database) ---
# We use a singleton check to prevent re-initialization errors on Vercel hot-reloads
if not firebase_admin._apps:
    # We will load credentials from Vercel Environment Variables
    # You must paste your Firebase JSON content into a Vercel Env Var named 'FIREBASE_CREDENTIALS'
    firebase_creds = os.environ.get('FIREBASE_CREDENTIALS')
    if firebase_creds:
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    else:
        print("⚠️ Warning: FIREBASE_CREDENTIALS env var not found.")

# Connect to DB
try:
    db = firestore.client()
except:
    db = None

# --- CONSTANTS ---
INITIAL_CAPITAL = 100000.0   
RISK_PER_TRADE = 0.02
COMMISSION_RATE = 0.001 
COLLECTION_NAME = "trading_bot"
DOC_NAME = "portfolio_state"

PORTFOLIO_CONFIG = {
    'WDC':  {'Entry': 50, 'Exit': 20}, 
    'STX':  {'Entry': 65, 'Exit': 35},
    'HOOD': {'Entry': 20, 'Exit': 10},
    'CAH':  {'Entry': 40, 'Exit': 30} 
}

# Standard Mean Reversion Settings
MR_SETTINGS = {'Window': 20, 'StdDev': 2.0} 

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_TOKEN')
TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID')

# --- HELPER FUNCTIONS ---

def send_telegram(message):
    if TELEGRAM_TOKEN and TELEGRAM_CHAT_ID:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
            requests.post(url, data=data, timeout=5)
        except: pass

def get_state():
    if db is None: return None
    doc_ref = db.collection(COLLECTION_NAME).document(DOC_NAME)
    doc = doc_ref.get()
    
    if doc.exists:
        return doc.to_dict()
    else:
        # Initialize default state in Cloud
        initial_state = {
            "cash": INITIAL_CAPITAL,
            "positions": {t: {"status": "NEUTRAL", "shares": 0, "entry_price": 0} for t in PORTFOLIO_CONFIG},
            "logs": [] 
        }
        doc_ref.set(initial_state)
        return initial_state

def save_state(state):
    if db is None: return
    # Limit logs to save space
    if len(state['logs']) > 50:
        state['logs'] = state['logs'][-50:]
    db.collection(COLLECTION_NAME).document(DOC_NAME).set(state)

def log_trade(state, ticker, strategy, action, price, shares, reason, balance):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{strategy}] {action} {ticker}: {shares:.2f} @ ${price:.2f} | Eq: ${balance:,.0f}"
    print(msg)
    send_telegram(f"🤖 Bot: {msg}\n({reason})")
    
    # Add to state logs instead of local CSV
    log_entry = f"{timestamp} | {ticker} | {action} | {price} | {shares} | {balance} | {reason}"
    state['logs'].append(log_entry)

def retry_download(ticker, lookback_days):
    try:
        # Reduced retries for serverless speed
        df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except: return None

def is_trading_hour():
    nyc = pytz.timezone('America/New_York')
    now = datetime.now(nyc)
    # Check Weekend (5=Sat, 6=Sun)
    if now.weekday() >= 5: return False
    # Check Hours (09:30 - 16:00)
    market_open = now.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now <= market_close

# --- INDICATORS ---
def get_atr(df, window=14):
    high, low, close = df['High'], df['Low'], df['Close'].shift(1)
    tr = pd.concat([high-low, (high-close).abs(), (low-close).abs()], axis=1).max(axis=1)
    return tr.rolling(window).mean().iloc[-1]

def get_adx(df, window=14):
    if len(df) < window * 2: return 0
    high, low, close = df['High'], df['Low'], df['Close']
    plus_dm, minus_dm = high.diff(), low.diff()
    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm > 0] = 0
    tr = pd.concat([high-low, (high-close.shift()).abs(), (low-close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/window, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1/window, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(alpha=1/window, adjust=False).mean().abs() / atr)
    dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
    return dx.ewm(alpha=1/window, adjust=False).mean().iloc[-1]

def calculate_position_size(price, volatility, total_equity):
    if volatility == 0 or price == 0: return 0
    risk_dollars = total_equity * RISK_PER_TRADE
    stop_distance = 2 * volatility
    if stop_distance == 0: return 0
    shares = risk_dollars / stop_distance
    max_position_val = total_equity * 0.24 
    return min(shares, max_position_val / price)

def get_current_equity(state, current_prices):
    equity = state['cash']
    for ticker, info in state['positions'].items():
        if info['shares'] > 0:
            price = current_prices.get(ticker, info['entry_price'])
            equity += (info['shares'] * price)
    return equity

# --- CORE LOGIC ---
def run_strategy_logic():
    if not is_trading_hour():
        return "Market Closed"

    state = get_state()
    if not state: return "Database Error"

    # 1. Update Portfolio Valuation
    current_prices = {}
    for ticker in PORTFOLIO_CONFIG:
        df = retry_download(ticker, 5)
        if df is not None: current_prices[ticker] = df['Close'].iloc[-1]
    
    total_equity = get_current_equity(state, current_prices)
    
    # 2. Execute Strategy per Ticker
    for ticker, config in PORTFOLIO_CONFIG.items():
        # Get History
        df = retry_download(ticker, 300)
        if df is None: continue
        
        price = df['Close'].iloc[-1]
        atr_val = get_atr(df)
        atr_pct = (atr_val / price) * 100
        adx = get_adx(df)
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        
        # --- STRATEGY SELECTION (Hybrid + Sniper) ---
        active_strategy = "MEAN_REV"
        
        # SQUEEZE: Volatility < 1% (Sniper Mode for CAH)
        if atr_pct < 1.0: 
            active_strategy = "SQUEEZE"
        # TURTLE: Strong Trend
        elif adx > 25 and price > sma_200: 
            active_strategy = "TURTLE"
        
        # --- GENERATE SIGNALS ---
        pos_data = state['positions'][ticker]
        current_status = pos_data['status']
        signal = "HOLD"
        reason = ""

        if active_strategy == "SQUEEZE":
            # Fast Breakout (20 Day High)
            hist_high_20 = df['High'].iloc[:-1].rolling(20).max().iloc[-1]
            if current_status == "NEUTRAL" and price > hist_high_20:
                signal, reason = "BUY", "Squeeze Breakout"
                
        elif active_strategy == "TURTLE":
            # Slow Breakout (Entry Day High)
            hist_high = df['High'].iloc[:-1].rolling(config['Entry']).max().iloc[-1]
            hist_low = df['Low'].iloc[:-1].rolling(config['Exit']).min().iloc[-1]
            if current_status == "NEUTRAL" and price > hist_high: signal, reason = "BUY", "Turtle Entry"
            elif current_status == "LONG" and price < hist_low: signal, reason = "SELL", "Turtle Exit"
            
        elif active_strategy == "MEAN_REV":
            # Buy Dips / Sell Mean
            sma = df['Close'].rolling(MR_SETTINGS['Window']).mean().iloc[-1]
            std = df['Close'].rolling(MR_SETTINGS['Window']).std().iloc[-1]
            lower = sma - (MR_SETTINGS['StdDev'] * std)
            if current_status == "NEUTRAL" and price < lower: signal, reason = "BUY", "Dip Buy"
            elif current_status == "LONG" and price > sma: signal, reason = "SELL", "Mean Revert"

        # --- EXECUTION ---
        if signal == "BUY" and state['cash'] > 0:
            shares = calculate_position_size(price, atr_val, total_equity)
            cost = (shares * price) * (1 + COMMISSION_RATE)
            if cost < state['cash'] and shares > 0:
                state['cash'] -= cost
                state['positions'][ticker] = {"status": "LONG", "shares": shares, "entry_price": price}
                log_trade(state, ticker, active_strategy, "BUY", price, shares, reason, total_equity)

        elif signal == "SELL" and current_status == "LONG":
            shares = pos_data['shares']
            revenue = (shares * price) * (1 - COMMISSION_RATE)
            state['cash'] += revenue
            state['positions'][ticker] = {"status": "NEUTRAL", "shares": 0, "entry_price": 0}
            log_trade(state, ticker, active_strategy, "SELL", price, shares, reason, total_equity)

    save_state(state)
    return "Logic Executed Successfully"

# --- VERCEL ROUTING ---
@app.route('/')
def home():
    return "QuantBot is Online. Send GET request to /run to trigger."

@app.route('/run')
def execute():
    try:
        result = run_strategy_logic()
        return jsonify({
            "status": "success",
            "message": result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)