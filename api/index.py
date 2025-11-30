from flask import Flask, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import firebase_admin
from firebase_admin import credentials, firestore
import os
import json
import warnings
import pytz
from datetime import datetime

# --- CONFIGURATION ---
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None

app = Flask(__name__)

# --- FIREBASE SETUP ---
if not firebase_admin._apps:
    firebase_creds = os.environ.get('FIREBASE_CREDENTIALS')
    if firebase_creds:
        cred_dict = json.loads(firebase_creds)
        cred = credentials.Certificate(cred_dict)
        firebase_admin.initialize_app(cred)
    else:
        print("⚠️ Warning: FIREBASE_CREDENTIALS env var not found.")

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

MR_SETTINGS = {'Window': 20, 'StdDev': 2.0} 

# --- HELPER FUNCTIONS ---

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
    # Keep only last 50 logs to save DB space
    if len(state['logs']) > 50:
        state['logs'] = state['logs'][-50:]
    db.collection(COLLECTION_NAME).document(DOC_NAME).set(state)

def log_trade(state, ticker, strategy, action, price, shares, reason, balance):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    msg = f"[{strategy}] {action} {ticker}: {shares:.2f} @ ${price:.2f} | Eq: ${balance:,.0f}"
    print(msg) 
    
    log_entry = f"{timestamp} | {ticker} | {action} | {price} | {shares} | {balance} | {reason}"
    state['logs'].append(log_entry)

def retry_download(ticker, lookback_days):
    try:
        df = yf.download(ticker, period=f"{lookback_days}d", interval="1d", progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        return df if not df.empty else None
    except: return None

def is_trading_hour():
    nyc = pytz.timezone('America/New_York')
    now = datetime.now(nyc)
    if now.weekday() >= 5: return False 
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
    
    plus_dm = high.diff()
    minus_dm = low.diff()
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
    # Uncomment the check below for production!
    # if not is_trading_hour():
    #     return "Market Closed"

    state = get_state()
    if not state: return "Database Error: Could not load state"

    current_prices = {}
    for ticker in PORTFOLIO_CONFIG:
        df = retry_download(ticker, 5)
        if df is not None: current_prices[ticker] = df['Close'].iloc[-1]
    
    total_equity = get_current_equity(state, current_prices)
    
    for ticker, config in PORTFOLIO_CONFIG.items():
        df = retry_download(ticker, 300)
        if df is None: continue
        
        price = df['Close'].iloc[-1]
        atr_val = get_atr(df)
        atr_pct = (atr_val / price) * 100
        adx = get_adx(df)
        sma_200 = df['Close'].rolling(200).mean().iloc[-1]
        
        active_strategy = "MEAN_REV"
        if atr_pct < 1.0: 
            active_strategy = "SQUEEZE"
        elif adx > 25 and price > sma_200: 
            active_strategy = "TURTLE"
        
        pos_data = state['positions'][ticker]
        current_status = pos_data['status']
        signal = "HOLD"
        reason = ""

        if active_strategy == "SQUEEZE":
            hist_high_20 = df['High'].iloc[:-1].rolling(20).max().iloc[-1]
            if current_status == "NEUTRAL" and price > hist_high_20:
                signal, reason = "BUY", "Squeeze Breakout"
                
        elif active_strategy == "TURTLE":
            hist_high = df['High'].iloc[:-1].rolling(config['Entry']).max().iloc[-1]
            hist_low = df['Low'].iloc[:-1].rolling(config['Exit']).min().iloc[-1]
            if current_status == "NEUTRAL" and price > hist_high: signal, reason = "BUY", "Turtle Entry"
            elif current_status == "LONG" and price < hist_low: signal, reason = "SELL", "Turtle Exit"
            
        elif active_strategy == "MEAN_REV":
            sma = df['Close'].rolling(MR_SETTINGS['Window']).mean().iloc[-1]
            std = df['Close'].rolling(MR_SETTINGS['Window']).std().iloc[-1]
            lower = sma - (MR_SETTINGS['StdDev'] * std)
            if current_status == "NEUTRAL" and price < lower: signal, reason = "BUY", "Dip Buy"
            elif current_status == "LONG" and price > sma: signal, reason = "SELL", "Mean Revert"

        if signal == "BUY" and state['cash'] > 0:
            shares = calculate_position_size(price, atr_val, total_equity)
            cost = (shares * price) * (1 + COMMISSION_RATE)
            
            if cost < state['cash'] and shares > 0:
                state['cash'] -= cost
                state['positions'][ticker] = {"status": "LONG", "shares": shares, "entry_price": price}
                log_trade(state, ticker, active_strategy, "BUY", price, shares, reason, total_equity)

        elif signal == "SELL" and current_status == "LONG":
            shares = pos_data['shares']
            rev = (shares * price) * (1 - COMMISSION_RATE)
            state['cash'] += rev
            state['positions'][ticker] = {"status": "NEUTRAL", "shares": 0, "entry_price": 0}
            log_trade(state, ticker, active_strategy, "SELL", price, shares, reason, total_equity)

    save_state(state)
    return "Logic Executed Successfully"

# --- VERCEL ROUTING ---

@app.route('/')
def home():
    """
    DISPLAY DASHBOARD
    Reads the state from Firebase and shows it as JSON in the browser.
    """
    try:
        state = get_state()
        if state:
            # Add a timestamp to know when you last checked
            state['server_time'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return jsonify(state)
        else:
            return jsonify({"status": "error", "message": "Database not connected. Check Env Vars."}), 500
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/run')
def execute():
    """
    TRIGGER BOT
    Called by cron-job.org to execute trading logic.
    """
    try:
        result = run_strategy_logic()
        return jsonify({
            "status": "success",
            "message": result,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Local testing
if __name__ == '__main__':
    app.run(debug=True)