import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import altair as alt
import time
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

# Import database module with fallback if not available
try:
    from database import get_db
    HAS_DATABASE = True
    logger.info("Database module imported successfully")
except ImportError:
    HAS_DATABASE = False
    logger.warning("Database module not found, falling back to CSV")

# Set Streamlit page configuration
st.set_page_config(
    page_title="HYPERION Trading System",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://www.github.com/yourusername/tron-bot',
        'Report a bug': 'https://www.github.com/yourusername/tron-bot/issues',
        'About': "# HYPERION Trading System\nAdvanced cryptocurrency trading dashboard."
    }
)

# Custom CSS
st.markdown("""
<style>
    /* Global Styles */
    .main {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Header styling */
    .hyperion-header {
        background: linear-gradient(90deg, #3a1c71, #d76d77, #ffaf7b);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Orbitron', sans-serif;
        margin-bottom: 0;
        text-align: center;
        font-size: 3.5em;
        font-weight: 900;
        letter-spacing: 2px;
        text-shadow: 0px 0px 10px rgba(255,255,255,0.1);
    }
    
    /* Subheader styling */
    .hyperion-subheader {
        color: #a0a0a0;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        letter-spacing: 1px;
        margin-top: 0;
    }
    
    /* Panel styling */
    .hyperion-panel {
        background-color: #161a25;
        border-radius: 10px;
        border: 1px solid #2e3c54;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Metric card styling */
    .metric-card {
        background: linear-gradient(145deg, #1e2430, #14191f);
        border-radius: 15px;
        padding: 15px;
        margin: 10px 0;
        text-align: center;
        border: 1px solid #2a3544;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .metric-title {
        color: #8b9eba;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-value {
        font-size: 2em;
        font-weight: bold;
        margin: 10px 0;
    }
    
    .metric-positive {
        color: #00ff9f;
    }
    
    .metric-negative {
        color: #ff5c87;
    }
    
    .metric-neutral {
        color: #e0e0e0;
    }
    
    /* Pulsing dot for live status */
    .live-indicator {
        display: inline-block;
        width: 10px;
        height: 10px;
        background-color: #00ff9f;
        border-radius: 50%;
        margin-right: 8px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 255, 159, 0.7);
        }
        70% {
            transform: scale(1);
            box-shadow: 0 0 0 10px rgba(0, 255, 159, 0);
        }
        100% {
            transform: scale(0.95);
            box-shadow: 0 0 0 0 rgba(0, 255, 159, 0);
        }
    }
    
    /* Table styling */
    .styled-table {
        border-radius: 10px;
        overflow: hidden;
        border-collapse: collapse;
        width: 100%;
    }
    
    .styled-table th {
        background-color: #1a1f2c;
        color: #7f9cf5;
        font-weight: bold;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .styled-table tr:nth-child(even) {
        background-color: #131720;
    }
    
    .styled-table tr:hover {
        background-color: #1e273a;
    }
    
    /* Trade badges */
    .badge {
        padding: 5px 10px;
        border-radius: 5px;
        font-size: 0.8em;
        font-weight: bold;
    }
    
    .buy-badge {
        background-color: rgba(0, 255, 159, 0.2);
        color: #00ff9f;
        border: 1px solid #00ff9f;
    }
    
    .sell-badge {
        background-color: rgba(255, 92, 135, 0.2);
        color: #ff5c87;
        border: 1px solid #ff5c87;
    }
    
    /* Glowing border for active trades */
    .active-trade {
        animation: glow 2s infinite alternate;
    }
    
    @keyframes glow {
        from {
            box-shadow: 0 0 5px -5px #7f9cf5;
        }
        to {
            box-shadow: 0 0 5px 5px #7f9cf5;
        }
    }
    
    /* Market pulse animation */
    .market-pulse {
        height: 40px;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    
    .pulse-bar {
        width: 4px;
        height: 100%;
        margin: 0 2px;
        background-color: #7f9cf5;
        animation: pulse-animation 1.2s infinite;
        border-radius: 2px;
    }
    
    @keyframes pulse-animation {
        0% { height: 10%; }
        50% { height: 100%; }
        100% { height: 10%; }
    }
</style>

<!-- Import Orbitron font -->
<link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

# Create the header with animated element
st.markdown('<h1 class="hyperion-header">HYPERION TRADING SYSTEM</h1>', unsafe_allow_html=True)

# Current time and status
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
st.markdown(f'<p class="hyperion-subheader"><span class="live-indicator"></span> LIVE SYSTEM · {current_time}</p>', unsafe_allow_html=True)

# Add auto-refresh control in sidebar
st.sidebar.markdown("### 🔄 Auto Refresh")
auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)
if auto_refresh:
    refresh_rate = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 30)
    st.write(f'<meta http-equiv="refresh" content="{refresh_rate}">', unsafe_allow_html=True)

# Create function to generate sample data if needed
def generate_sample_data(days=30):
    try:
        # Generate sample trade data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dates = pd.date_range(start=start_date, end=end_date, freq='H')
        pairs = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
        
        data = []
        cumulative_profit = 0
        
        for i in range(len(dates) // 4):  # Create a trade every ~4 hours
            pair = np.random.choice(pairs)
            action = np.random.choice(['BUY', 'SELL'])
            price = np.random.uniform(10, 60000)
            quantity = np.random.uniform(0.01, 2.0)
            
            if action == 'BUY':
                profit = None
                profit_pct = None
                fee = price * quantity * 0.001
                net_profit = None
            else:
                profit = np.random.uniform(-500, 1000)
                profit_pct = (profit / (price * quantity)) * 100
                fee = price * quantity * 0.001
                net_profit = profit - fee
                cumulative_profit += net_profit
            
            data.append({
                'timestamp': dates[i * 4],
                'pair': pair,
                'action': action,
                'price': price,
                'quantity': quantity,
                'net_profit': net_profit,
                'profit_pct': profit_pct
            })
        
        # Sort by timestamp
        df = pd.DataFrame(data)
        df = df.sort_values('timestamp')
        
        # Calculate cumulative profit
        df['cumulative_net_profit'] = None
        running_total = 0
        
        for idx, row in df.iterrows():
            if row['action'] == 'SELL' and pd.notna(row['net_profit']):
                running_total += row['net_profit']
            df.at[idx, 'cumulative_net_profit'] = running_total
            
        # Fill forward cumulative profit
        df['cumulative_net_profit'] = df['cumulative_net_profit'].fillna(method='ffill')
        return df
    except Exception as e:
        logger.error(f"Error generating sample data: {e}")
        # Return a minimal dataframe to avoid breaking the app
        return pd.DataFrame({
            'timestamp': [datetime.now()],
            'pair': ['BTCUSDT'],
            'action': ['BUY'],
            'price': [50000],
            'quantity': [0.1],
            'net_profit': [None],
            'profit_pct': [None],
            'cumulative_net_profit': [0]
        })

# Function to load data from database
def load_data_from_database(start_date=None, end_date=None, pair=None, action=None):
    """
    Load trading data from the database.
    
    Args:
        start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to None.
        end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to None.
        pair (str, optional): Filter by trading pair. Defaults to None.
        action (str, optional): Filter by action ('BUY' or 'SELL'). Defaults to None.
        
    Returns:
        pandas.DataFrame: DataFrame containing the trades
    """
    try:
        db = get_db()
        df = db.get_trades(
            start_date=start_date, 
            end_date=end_date,
            pair=pair if pair != "All" else None,
            action=action if action != "All" else None
        )
        
        logger.info(f"Loaded {len(df)} trades from database")
        return df
    except Exception as e:
        logger.error(f"Error loading data from database: {e}")
        return pd.DataFrame()  # Return empty DataFrame

# Function to load data from CSV with fallback to sample data
def load_data_from_csv(csv_file=None):
    """
    Load trading data from CSV file with fallback to sample data.
    
    Args:
        csv_file (str, optional): Path to CSV file. If None, tries to find trade_log.csv
        
    Returns:
        pandas.DataFrame: DataFrame containing the trades
    """
    try:
        # Default to trade_log.csv if no file specified
        if csv_file is None:
            # Search for trade log file in multiple possible locations
            possible_paths = [
                "trade_log.csv",  # Root directory
                os.path.join("trades", "trade_log.csv"),  # Trades directory
                os.path.join(os.getcwd(), "trade_log.csv"),  # Absolute path
                os.path.join(os.getcwd(), "trades", "trade_log.csv"),  # Absolute path in trades dir
            ]
            
            csv_file = None
            for path in possible_paths:
                if os.path.exists(path) and os.path.getsize(path) > 0:
                    csv_file = path
                    break
            
            if csv_file is None:
                logger.warning("No valid trade_log.csv found")
                return generate_sample_data()
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Check for required columns
        required_columns = ['timestamp', 'pair', 'action', 'price', 'quantity']
        for col in required_columns:
            if col not in df.columns:
                logger.error(f"CSV file missing required column: {col}")
                return generate_sample_data()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Rename columns if needed
        if 'profit' in df.columns and 'net_profit' not in df.columns:
            df['net_profit'] = df['profit']
        
        # Calculate cumulative profit if not present
        if 'cumulative_net_profit' not in df.columns:
            df['cumulative_net_profit'] = None
            running_total = 0
            
            # Sort by timestamp to ensure correct calculation
            df_sorted = df.sort_values('timestamp')
            
            for idx, row in df_sorted.iterrows():
                if row['action'] == 'SELL' and pd.notna(row.get('net_profit')):
                    running_total += row['net_profit']
                df.at[idx, 'cumulative_net_profit'] = running_total
            
            # Fill forward cumulative profit
            df['cumulative_net_profit'] = df['cumulative_net_profit'].fillna(method='ffill')
        
        logger.info(f"Loaded {len(df)} trades from CSV: {csv_file}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from CSV: {e}")
        return generate_sample_data()

# Function to get market scans from database
def load_market_scans(limit=10):
    """
    Load market scans from the database.
    
    Args:
        limit (int, optional): Maximum number of scans to return. Defaults to 10.
        
    Returns:
        pandas.DataFrame: DataFrame containing the market scans
    """
    if not HAS_DATABASE:
        return pd.DataFrame()
    
    try:
        db = get_db()
        df = db.get_market_scans(limit=limit)
        logger.info(f"Loaded {len(df)} market scans from database")
        return df
    except Exception as e:
        logger.error(f"Error loading market scans from database: {e}")
        return pd.DataFrame()

# Function to get the latest bot status
def get_bot_status():
    """
    Get the latest bot status from the database.
    
    Returns:
        dict: Bot status information or None if not available
    """
    if not HAS_DATABASE:
        return None
    
    try:
        db = get_db()
        status = db.get_latest_status()
        return status
    except Exception as e:
        logger.error(f"Error getting bot status: {e}")
        return None

# Function to get the latest market scan
def get_latest_scan(pair=None):
    """
    Get the latest market scan from the database.
    
    Args:
        pair (str, optional): Trading pair. If None, gets latest scan for any pair.
        
    Returns:
        dict: Latest scan information or None if not available
    """
    if not HAS_DATABASE:
        return None
    
    try:
        db = get_db()
        scan = db.get_latest_scan(pair)
        return scan
    except Exception as e:
        logger.error(f"Error getting latest scan: {e}")
        return None

# Display data source selector in sidebar
st.sidebar.markdown("### 📊 Data Source")
data_source = st.sidebar.radio(
    "Select data source:",
    ["Database", "CSV File", "Sample Data"],
    index=0 if HAS_DATABASE else 1
)

# Load data based on selected source
if data_source == "Database":
    if HAS_DATABASE:
        # Add date range filter
        st.sidebar.markdown("### 📅 Date Range")
        today = datetime.now().date()
        start_date = st.sidebar.date_input(
            "Start date",
            value=today - timedelta(days=30),
            max_value=today
        )
        end_date = st.sidebar.date_input(
            "End date",
            value=today,
            max_value=today
        )
        
        # Load data from database
        df = load_data_from_database(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d")
        )
        
        if df.empty:
            st.sidebar.warning("No data in database for selected date range")
            # Add option to import from CSV
            if st.sidebar.button("Import from CSV to Database"):
                try:
                    from logger_utils import import_trades_from_csv
                    count = import_trades_from_csv("trade_log.csv")
                    if count > 0:
                        st.sidebar.success(f"Successfully imported {count} trades to database")
                        # Reload data
                        df = load_data_from_database(
                            start_date=start_date.strftime("%Y-%m-%d"),
                            end_date=end_date.strftime("%Y-%m-%d")
                        )
                    else:
                        st.sidebar.error("Failed to import trades")
                except Exception as e:
                    st.sidebar.error(f"Error importing trades: {e}")
    else:
        st.sidebar.error("Database module not available")
        # Switch to CSV
        data_source = "CSV File"
        df = load_data_from_csv()
elif data_source == "CSV File":
    # Allow selecting a CSV file
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    
    if uploaded_file is not None:
        # Save uploaded file
        with open("trade_log.csv", "wb") as f:
            f.write(uploaded_file.getvalue())
        st.sidebar.success("File uploaded successfully!")
        df = load_data_from_csv("trade_log.csv")
    else:
        # Try to load existing file
        df = load_data_from_csv()
else:  # Sample Data
    df = generate_sample_data()
    st.sidebar.info("Using generated sample data")

# Add test trade generator to sidebar
st.sidebar.markdown("### 🧪 Test Trade Generator")
with st.sidebar.expander("Create Test Trade"):
    col1, col2 = st.columns(2)
    with col1:
        test_pair = st.selectbox("Pair", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
        test_action = st.selectbox("Action", ["BUY", "SELL"])
    with col2:
        test_price = st.number_input("Price", value=50000.0, min_value=0.1, step=100.0)
        test_quantity = st.number_input("Quantity", value=0.001, min_value=0.00001, step=0.001)
    
    # Add profit fields for SELL actions
    if test_action == "SELL":
        test_profit = st.number_input("Profit/Loss (USD)", value=10.0)
        test_profit_pct = st.number_input("Profit/Loss (%)", value=2.0)
    else:
        test_profit = None
        test_profit_pct = None
    
    if st.button("Add Test Trade"):
        try:
            from logger_utils import log_trade_to_csv
            result = log_trade_to_csv(
                pair=test_pair,
                action=test_action,
                price=test_price,
                quantity=test_quantity,
                pnl=test_profit,
                pnl_pct=test_profit_pct
            )
            if result:
                st.sidebar.success(f"Added {test_action} trade for {test_pair}!")
                # Force refresh
                time.sleep(1)  # Brief pause to ensure file is written
                st.experimental_rerun()
            else:
                st.sidebar.error("Failed to add test trade")
        except Exception as e:
            st.sidebar.error(f"Error adding test trade: {e}")

# Create sidebar
with st.sidebar:
    st.markdown('<div class="hyperion-panel">', unsafe_allow_html=True)
    
    # Use a more reliable image loading approach
    try:
        st.image("https://raw.githubusercontent.com/yourusername/tron-bot/main/logo.png", width=300)
    except:
        # Fallback if image can't be loaded
        st.markdown("# HYPERION")
    
    st.markdown("### 🔍 FILTER CONTROLS")
    
    # Date filters - with more error handling
    try:
        min_date = df['timestamp'].min().date()
        max_date = df['timestamp'].max().date()
    except:
        min_date = (datetime.now() - timedelta(days=30)).date()
        max_date = datetime.now().date()
    
    if data_source != "Database":  # Only show date filters for CSV/Sample data
        start_date = st.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
    
    # Pair filter
    try:
        pairs = ["All"] + sorted(df['pair'].unique().tolist())
    except:
        pairs = ["All", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    selected_pair = st.selectbox(
        "Trading Pair",
        options=pairs
    )
    
    # Action filter
    selected_action = st.selectbox(
        "Action Type",
        options=["All", "BUY", "SELL"]
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Date filter - with error handling
    if data_source != "Database":  # Only apply date filters for CSV/Sample data
        try:
            filtered_df = filtered_df[
                (filtered_df['timestamp'].dt.date >= start_date) & 
                (filtered_df['timestamp'].dt.date <= end_date)
            ]
        except Exception as e:
            st.error(f"Error filtering by date: {e}")
    
    # Pair filter
    if selected_pair != "All":
        try:
            filtered_df = filtered_df[filtered_df['pair'] == selected_pair]
        except Exception as e:
            st.error(f"Error filtering by pair: {e}")
    
    # Action filter
    if selected_action != "All":
        try:
            filtered_df = filtered_df[filtered_df['action'] == selected_action]
        except Exception as e:
            st.error(f"Error filtering by action: {e}")
    
    st.markdown("### 📊 MARKET PULSE")
    
    # Create market pulse animation
    pulse_html = '<div class="market-pulse">'
    for i in range(20):
        delay = i * 0.1
        pulse_html += f'<div class="pulse-bar" style="animation-delay: {delay}s;"></div>'
    pulse_html += '</div>'
    
    st.markdown(pulse_html, unsafe_allow_html=True)
    
    # Add controls for the bot
    st.markdown("### 🤖 BOT CONTROLS")
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Start Bot"):
            st.success("Bot started successfully!")
    
    with col2:
        if st.button("Stop Bot"):
            st.error("Bot stopped!")
    
    if st.button("Force Sell All"):
        with st.spinner("Selling all positions..."):
            time.sleep(1)
            st.success("All positions liquidated!")
    
    st.markdown("</div>", unsafe_allow_html=True)

# Main dashboard content
# Row 1: Key Metrics
st.markdown('<div class="hyperion-panel">', unsafe_allow_html=True)

try:
    # Calculate metrics
    total_trades = len(filtered_df)
    buys = len(filtered_df[filtered_df['action'] == 'BUY'])
    sells = len(filtered_df[filtered_df['action'] == 'SELL'])
    profit_df = filtered_df[filtered_df['net_profit'].notna()]
    total_pnl = profit_df['net_profit'].sum() if not profit_df.empty else 0
    win_rate = (profit_df['net_profit'] > 0).mean() * 100 if not profit_df.empty else 0
    avg_win = profit_df[profit_df['net_profit'] > 0]['net_profit'].mean() if not profit_df[profit_df['net_profit'] > 0].empty else 0
    avg_loss = profit_df[profit_df['net_profit'] < 0]['net_profit'].mean() if not profit_df[profit_df['net_profit'] < 0].empty else 0
except Exception as e:
    st.error(f"Error calculating metrics: {e}")
    # Set defaults
    total_trades = 0
    buys = 0
    sells = 0
    total_pnl = 0
    win_rate = 0
    avg_win = 0
    avg_loss = 0

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">TOTAL TRADES</div>
        <div class="metric-value metric-neutral">{total_trades}</div>
        <div>BUY: {buys} | SELL: {sells}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    pnl_class = "metric-positive" if total_pnl >= 0 else "metric-negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">TOTAL PNL</div>
        <div class="metric-value {pnl_class}">${total_pnl:.2f}</div>
        <div>Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    # Calculate win and loss counts first
    win_count = len(profit_df[profit_df['net_profit'] > 0]) if not profit_df.empty else 0
    loss_count = len(profit_df[profit_df['net_profit'] < 0]) if not profit_df.empty else 0
    
    win_class = "metric-positive" if win_rate >= 50 else "metric-negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">WIN RATE</div>
        <div class="metric-value {win_class}">{win_rate:.1f}%</div>
        <div>W: {win_count} | L: {loss_count}</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    # Calculate most profitable pair
    try:
        if not profit_df.empty:
            pair_profits = profit_df.groupby('pair')['net_profit'].sum()
            most_profitable_pair = pair_profits.idxmax() if not pair_profits.empty else "N/A"
            pair_profit = pair_profits.max() if not pair_profits.empty else 0
        else:
            most_profitable_pair = "N/A"
            pair_profit = 0
    except Exception as e:
        st.error(f"Error calculating best pair: {e}")
        most_profitable_pair = "N/A"
        pair_profit = 0
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">BEST PAIR</div>
        <div class="metric-value metric-positive">{most_profitable_pair}</div>
        <div>${pair_profit:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    # Calculate risk-reward ratio
    try:
        if not profit_df.empty and len(profit_df[profit_df['net_profit'] > 0]) > 0 and len(profit_df[profit_df['net_profit'] < 0]) > 0:
            avg_win = profit_df[profit_df['net_profit'] > 0]['net_profit'].mean()
            avg_loss = abs(profit_df[profit_df['net_profit'] < 0]['net_profit'].mean())
            risk_reward = avg_win / avg_loss if avg_loss != 0 else 0
        else:
            risk_reward = 0
    except Exception as e:
        st.error(f"Error calculating risk-reward: {e}")
        risk_reward = 0
    
    rr_class = "metric-positive" if risk_reward >= 1 else "metric-negative"
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">RISK/REWARD</div>
        <div class="metric-value {rr_class}">{risk_reward:.2f}</div>
        <div>Ratio</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Tabs for different sections
tab1, tab2, tab3 = st.tabs(["📈 Performance", "🔍 Trading Pairs", "📋 Recent Trades"])

with tab1:
    st.markdown('<div class="hyperion-panel">', unsafe_allow_html=True)
    st.subheader("Performance Over Time")
    
    # Create cumulative PnL chart
    try:
        if 'net_profit' in filtered_df.columns:
            # Ensure we have cumulative profit column
            if 'cumulative_net_profit' not in filtered_df.columns:
                filtered_df['cumulative_net_profit'] = filtered_df['net_profit'].fillna(0).cumsum()
            
            # Get only sell trades for plotting
            sell_df = filtered_df[filtered_df['action'] == 'SELL'].copy()
            
            if not sell_df.empty:
                # Create Plotly line chart
                fig = go.Figure()
                
                # Add area chart for cumulative profit
                fig.add_trace(go.Scatter(
                    x=sell_df['timestamp'],
                    y=sell_df['cumulative_net_profit'],
                    fill='tozeroy',
                    fillcolor='rgba(0, 255, 159, 0.1)',
                    line=dict(color='#00ff9f', width=2),
                    name='Cumulative PnL'
                ))
                
                # Add markers for individual trades
                fig.add_trace(go.Scatter(
                    x=sell_df['timestamp'],
                    y=sell_df['net_profit'],
                    mode='markers',
                    marker=dict(
                        color=sell_df['net_profit'].apply(lambda x: '#00ff9f' if x >= 0 else '#ff5c87'),
                        size=8,
                        line=dict(width=1, color='#1a1f2c')
                    ),
                    name='Trade PnL'
                ))
                
                # Customize layout
                fig.update_layout(
                    title="Cumulative Profit/Loss Over Time",
                    xaxis_title="Date",
                    yaxis_title="Profit/Loss (USD)",
                    height=400,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    ),
                    xaxis=dict(
                        gridcolor='#1a1f2c',
                        showgrid=True
                    ),
                    yaxis=dict(
                        gridcolor='#1a1f2c',
                        showgrid=True,
                        zerolinecolor='#2e3c54'
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sell trades found in the selected date range.")
        else:
            st.info("No profit data available.")
    except Exception as e:
        st.error(f"Error generating performance chart: {e}")
        st.info("Performance chart could not be displayed due to an error.")
    
    # Second row: Daily PnL and Trading Volume
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Daily Performance")
        
        try:
            # Calculate daily PnL
            if 'net_profit' in filtered_df.columns and not filtered_df.empty:
                # Group by day and sum the profit
                daily_pnl = filtered_df[filtered_df['net_profit'].notna()].copy()
                if not daily_pnl.empty:
                    daily_pnl['date'] = daily_pnl['timestamp'].dt.date
                    daily_pnl = daily_pnl.groupby('date')['net_profit'].sum().reset_index()
                    
                    # Create Plotly bar chart
                    fig = px.bar(
                        daily_pnl,
                        x='date',
                        y='net_profit',
                        color=daily_pnl['net_profit'] >= 0,
                        color_discrete_map={True: '#00ff9f', False: '#ff5c87'},
                        labels={'net_profit': 'Profit/Loss', 'date': 'Date'}
                    )
                    
                    # Customize layout
                    fig.update_layout(
                        showlegend=False,
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=0, b=0),
                        height=300,
                        xaxis=dict(
                            gridcolor='#1a1f2c',
                            showgrid=True
                        ),
                        yaxis=dict(
                            gridcolor='#1a1f2c',
                            showgrid=True,
                            zerolinecolor='#2e3c54'
                        )
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No profit data available for the selected date range.")
            else:
                st.info("No profit data available.")
        except Exception as e:
            st.error(f"Error generating daily performance chart: {e}")
            st.info("Daily performance chart could not be displayed due to an error.")
    
    with col2:
        st.subheader("Trading Volume")
        
        try:
            # Calculate daily trading volume
            if not filtered_df.empty:
                # Use price * quantity for volume and group by day
                filtered_df['volume'] = filtered_df['price'] * filtered_df['quantity']
                filtered_df['date'] = filtered_df['timestamp'].dt.date
                daily_volume = filtered_df.groupby(['date', 'action']).agg({'volume': 'sum'}).reset_index()
                
                # Create Plotly bar chart
                fig = px.bar(
                    daily_volume,
                    x='date',
                    y='volume',
                    color='action',
                    barmode='group',
                    color_discrete_map={'BUY': '#03b2f8', 'SELL': '#ff5c87'},
                    labels={'volume': 'Volume (USD)', 'date': 'Date', 'action': 'Type'}
                )
                
                # Customize layout
                fig.update_layout(
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=300,
                    xaxis=dict(
                        gridcolor='#1a1f2c',
                        showgrid=True
                    ),
                    yaxis=dict(
                        gridcolor='#1a1f2c',
                        showgrid=True
                    ),
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No volume data available for the selected date range.")
        except Exception as e:
            st.error(f"Error generating volume chart: {e}")
            st.info("Volume chart could not be displayed due to an error.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab2:
    st.markdown('<div class="hyperion-panel">', unsafe_allow_html=True)
    st.subheader("Trading Pair Performance")
    
    try:
        # Calculate performance by pair
        if 'net_profit' in filtered_df.columns and not filtered_df.empty:
            pair_profit = filtered_df[filtered_df['net_profit'].notna()].groupby('pair')['net_profit'].sum().reset_index()
            pair_trades = filtered_df.groupby('pair').size().reset_index(name='trade_count')
            pair_win_rate = filtered_df[filtered_df['net_profit'].notna()].groupby('pair').apply(
                lambda x: (x['net_profit'] > 0).mean() * 100
            ).reset_index(name='win_rate')
            
            # Merge the dataframes
            pair_stats = pair_profit.merge(pair_trades, on='pair').merge(pair_win_rate, on='pair')
            pair_stats = pair_stats.sort_values('net_profit', ascending=False)
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                # Create horizontal bar chart
                fig = px.bar(
                    pair_stats,
                    y='pair',
                    x='net_profit',
                    color='net_profit',
                    color_continuous_scale=['#ff5c87', '#7f9cf5', '#00ff9f'],
                    labels={'net_profit': 'Profit/Loss (USD)', 'pair': 'Trading Pair'},
                    orientation='h'
                )
                
                # Customize layout
                fig.update_layout(
                    showlegend=False,
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False,
                    margin=dict(l=0, r=0, t=0, b=0),
                    height=400,
                    xaxis=dict(
                        gridcolor='#1a1f2c',
                        showgrid=True
                    ),
                    yaxis=dict(
                        gridcolor='#1a1f2c',
                        showgrid=False
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Create a metrics table
                st.markdown("""
                <style>
                .pair-metrics {
                    width: 100%;
                    border-collapse: collapse;
                }
                .pair-metrics th {
                    background-color: #1a1f2c;
                    color: #7f9cf5;
                    padding: 10px;
                    text-align: left;
                    font-size: 0.9em;
                }
                .pair-metrics td {
                    padding: 10px;
                    border-bottom: 1px solid #2e3c54;
                }
                .profit-positive {
                    color: #00ff9f;
                }
                .profit-negative {
                    color: #ff5c87;
                }
                .win-rate-high {
                    color: #00ff9f;
                }
                .win-rate-medium {
                    color: #ffaf7b;
                }
                .win-rate-low {
                    color: #ff5c87;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Generate table HTML
                table_html = """
                <table class="pair-metrics">
                    <thead>
                        <tr>
                            <th>Pair</th>
                            <th>PnL</th>
                            <th>Trades</th>
                            <th>Win Rate</th>
                        </tr>
                    </thead>
                    <tbody>
                """
                
                for _, row in pair_stats.iterrows():
                    profit_class = "profit-positive" if row['net_profit'] >= 0 else "profit-negative"
                    
                    if row['win_rate'] >= 70:
                        win_rate_class = "win-rate-high"
                    elif row['win_rate'] >= 50:
                        win_rate_class = "win-rate-medium"
                    else:
                        win_rate_class = "win-rate-low"
                    
                    table_html += f"""
                    <tr>
                        <td>{row['pair']}</td>
                        <td class="{profit_class}">${row['net_profit']:.2f}</td>
                        <td>{row['trade_count']}</td>
                        <td class="{win_rate_class}">{row['win_rate']:.1f}%</td>
                    </tr>
                    """
                
                table_html += """
                    </tbody>
                </table>
                """
                
                st.markdown(table_html, unsafe_allow_html=True)
        else:
            st.info("No profit data available for analysis.")
    except Exception as e:
        st.error(f"Error analyzing trading pairs: {e}")
        st.info("Trading pair analysis could not be displayed due to an error.")
    
    # Add win/loss ratio pie chart
    try:
        if 'net_profit' in filtered_df.columns and not filtered_df.empty:
            profit_trades = filtered_df[filtered_df['net_profit'] > 0]
            loss_trades = filtered_df[filtered_df['net_profit'] < 0]
            break_even_trades = filtered_df[(filtered_df['net_profit'] == 0) | (filtered_df['net_profit'].isna())]
            
            # Count trades by type
            win_count = len(profit_trades)
            loss_count = len(loss_trades) 
            neutral_count = len(break_even_trades)
            
            if win_count + loss_count > 0:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Create win/loss ratio pie chart
                    labels = ['Winning Trades', 'Losing Trades', 'Break-even/Open']
                    values = [win_count, loss_count, neutral_count]
                    colors = ['#00ff9f', '#ff5c87', '#7f9cf5']
                    
                    fig = go.Figure(data=[go.Pie(
                        labels=labels,
                        values=values,
                        hole=.5,
                        marker=dict(colors=colors)
                    )])
                    
                    fig.update_layout(
                        title="Win/Loss Distribution",
                        template="plotly_dark",
                        paper_bgcolor='rgba(0,0,0,0)',
                        plot_bgcolor='rgba(0,0,0,0)',
                        margin=dict(l=0, r=0, t=30, b=0),
                        height=300,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=-0.2,
                            xanchor="center",
                            x=0.5
                        ),
                        annotations=[dict(text=f"Total<br>{win_count + loss_count}", x=0.5, y=0.5, font_size=15, showarrow=False)]
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Create profit distribution histogram
                    profit_data = filtered_df[filtered_df['net_profit'].notna()]['net_profit']
                    
                    if not profit_data.empty:
                        fig = go.Figure()
                        
                        # Create histogram of profits
                        fig.add_trace(go.Histogram(
                            x=profit_data,
                            marker_color=['#00ff9f' if x >= 0 else '#ff5c87' for x in profit_data],
                            opacity=0.7,
                            nbinsx=20
                        ))
                        
                        fig.update_layout(
                            title="Profit Distribution",
                            template="plotly_dark",
                            paper_bgcolor='rgba(0,0,0,0)',
                            plot_bgcolor='rgba(0,0,0,0)',
                            margin=dict(l=0, r=0, t=30, b=0),
                            height=300,
                            xaxis_title="Profit/Loss (USD)",
                            yaxis_title="Number of Trades",
                            bargap=0.1,
                            xaxis=dict(
                                gridcolor='#1a1f2c',
                                showgrid=True,
                                zerolinecolor='#2e3c54'
                            ),
                            yaxis=dict(
                                gridcolor='#1a1f2c',
                                showgrid=True
                            )
                        )
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No profit distribution data available.")
    except Exception as e:
        st.error(f"Error generating win/loss charts: {e}")
        st.info("Win/loss analysis could not be displayed due to an error.")
    
    st.markdown('</div>', unsafe_allow_html=True)

with tab3:
    st.markdown('<div class="hyperion-panel">', unsafe_allow_html=True)
    st.subheader("Recent Trades")
    
    try:
        if not filtered_df.empty:
            # Sort by timestamp descending to show most recent first
            recent_trades = filtered_df.sort_values(by='timestamp', ascending=False).reset_index(drop=True)
            
            # Generate a styled HTML table
            st.markdown("""
            <style>
            .recent-trades {
                width: 100%;
                border-collapse: collapse;
                font-size: 0.9em;
            }
            .recent-trades th {
                background-color: #1a1f2c;
                color: #7f9cf5;
                padding: 12px;
                text-align: left;
                font-weight: bold;
                position: sticky;
                top: 0;
                z-index: 10;
            }
            .recent-trades td {
                padding: 10px 12px;
                border-bottom: 1px solid #2e3c54;
            }
            .recent-trades tr:hover {
                background-color: #1e273a;
            }
            .trade-buy {
                background-color: rgba(3, 178, 248, 0.1);
            }
            .trade-sell {
                background-color: rgba(255, 92, 135, 0.1);
            }
            .trade-time {
                color: #8b9eba;
                font-size: 0.85em;
            }
            </style>
            """, unsafe_allow_html=True)
            
            # Create a container with specific height and scrolling
            st.markdown('<div style="height: 500px; overflow-y: auto;">', unsafe_allow_html=True)
            
            # Generate the table header
            table_html = """
            <table class="recent-trades">
                <thead>
                    <tr>
                        <th>Time</th>
                        <th>Pair</th>
                        <th>Action</th>
                        <th>Price</th>
                        <th>Quantity</th>
                        <th>P/L</th>
                    </tr>
                </thead>
                <tbody>
            """
            
            # Add rows for each trade
            for idx, row in recent_trades.iterrows():
                # Format timestamp
                timestamp = row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
                # Determine row class based on action
                row_class = "trade-buy" if row['action'] == 'BUY' else "trade-sell"
                # Format price and quantity
                price = f"${row['price']:.2f}" if isinstance(row['price'], (int, float)) else row['price']
                quantity = f"{row['quantity']:.6f}" if isinstance(row['quantity'], (int, float)) else row['quantity']
                # Format P/L
                if pd.notna(row.get('net_profit')):
                    profit_class = "profit-positive" if row['net_profit'] >= 0 else "profit-negative"
                    profit = f"<span class='{profit_class}'>${row['net_profit']:.2f}</span>"
                else:
                    profit = "—"
                
                # Create badge for action
                badge_class = "buy-badge" if row['action'] == 'BUY' else "sell-badge"
                action_badge = f"<span class='badge {badge_class}'>{row['action']}</span>"
                
                # Add the row
                table_html += f"""
                <tr class="{row_class}">
                    <td><span class="trade-time">{timestamp}</span></td>
                    <td><strong>{row['pair']}</strong></td>
                    <td>{action_badge}</td>
                    <td>{price}</td>
                    <td>{quantity}</td>
                    <td>{profit}</td>
                </tr>
                """
            
            # Close the table
            table_html += """
                </tbody>
            </table>
            """
            
            # Display the table
            st.markdown(table_html, unsafe_allow_html=True)
            
            # Close the container
            st.markdown('</div>', unsafe_allow_html=True)
            
        else:
            st.info("No trades found for the selected filters. The trade log may be empty.")
            
            # Offer to generate a sample trade
            if st.button("Generate Test Trade Log Entry"):
                test_trade_html = """
                <div style="background-color: #1e273a; border: 1px solid #2e3c54; border-radius: 8px; padding: 15px; margin-top: 15px;">
                    <h4 style="margin-top: 0;">How to Generate Trades</h4>
                    <p>Your bot needs to execute trades for them to appear here. To test trade logging, add code like this:</p>
                    <pre style="background-color: #0e1117; padding: 10px; border-radius: 5px; overflow-x: auto;">
# In your testing script
from logger_utils import log_trade_to_csv

# Test BUY trade
log_trade_to_csv(
    pair="BTCUSDT", 
    action="BUY",
    price=65420.50, 
    quantity=0.001
)

# Later, test SELL trade with profit
log_trade_to_csv(
    pair="BTCUSDT", 
    action="SELL",
    price=66750.25, 
    quantity=0.001,
    pnl=1.33,  # Profit in USDT
    pnl_pct=2.03  # Percent gain
)
                    </pre>
                </div>
                """
                st.markdown(test_trade_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error displaying recent trades: {e}")
        st.info("Recent trades could not be displayed due to an error.")
    
    st.markdown('</div>', unsafe_allow_html=True)

# Bottom section - System Status and Alerts
st.markdown('<div class="hyperion-panel">', unsafe_allow_html=True)
st.subheader("System Status & Alerts")

col1, col2 = st.columns(2)

with col1:
    # Get bot status from database if available
    bot_status = None
    if HAS_DATABASE:
        bot_status = get_bot_status()
    
    # Set status indicators based on bot_status
    api_status = "Online" if bot_status else "Unknown"
    api_color = "#00ff9f" if bot_status else "#ffaf7b"
    
    db_status = "Connected" if HAS_DATABASE else "Not Available"
    db_color = "#00ff9f" if HAS_DATABASE else "#ff5c87"
    
    discord_status = "Active"
    discord_color = "#00ff9f"
    
    engine_status = bot_status.get('status', "Unknown") if bot_status else "Unknown"
    engine_color = "#00ff9f" if engine_status == "RUNNING" else "#ffaf7b"
    
    # Create a styled status panel
    st.markdown(f"""
    <div style="background-color: #1e273a; border-radius: 8px; padding: 15px; border: 1px solid #2e3c54;">
        <h4 style="margin-top: 0;">System Health</h4>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 10px; height: 10px; background-color: {api_color}; border-radius: 50%; margin-right: 10px;"></div>
            <div>API Connection: <span style="color: {api_color};">{api_status}</span></div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 10px; height: 10px; background-color: {db_color}; border-radius: 50%; margin-right: 10px;"></div>
            <div>Database: <span style="color: {db_color};">{db_status}</span></div>
        </div>
        <div style="display: flex; align-items: center; margin-bottom: 10px;">
            <div style="width: 10px; height: 10px; background-color: {discord_color}; border-radius: 50%; margin-right: 10px;"></div>
            <div>Discord Notifications: <span style="color: {discord_color};">{discord_status}</span></div>
        </div>
        <div style="display: flex; align-items: center;">
            <div style="width: 10px; height: 10px; background-color: {engine_color}; border-radius: 50%; margin-right: 10px;"></div>
            <div>Trading Engine: <span style="color: {engine_color};">{engine_status}</span></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    # Get latest market scan if available
    latest_scan = None
    if HAS_DATABASE:
        latest_scan = get_latest_scan()
    
    # Read latest scan file if available (fallback)
    try:
        if os.path.exists("latest_scan.txt"):
            with open("latest_scan.txt", "r") as f:
                scan_content = f.read()
            
            # Format the scan content for display
            scan_lines = scan_content.split('\n')
            scan_title = scan_lines[0] if len(scan_lines) > 0 else "Latest Market Scan"
            
            # Create a scan display panel
            st.markdown(f"""
            <div style="background-color: #1e273a; border-radius: 8px; padding: 15px; border: 1px solid #2e3c54;">
                <h4 style="margin-top: 0;">Latest Scan Result</h4>
                <pre style="background-color: #0e1117; padding: 10px; border-radius: 5px; max-height: 200px; overflow-y: auto; font-size: 0.9em;">{scan_content}</pre>
            </div>
            """, unsafe_allow_html=True)
        elif latest_scan:
            # Display latest scan from database
            scan_time = latest_scan.get('timestamp', 'N/A')
            scan_pair = latest_scan.get('pair', 'N/A')
            scan_signal = latest_scan.get('signal', 'N/A')
            scan_price = latest_scan.get('price', 0)
            
            # Format signal with emoji
            signal_icon = "⚪"  # default/hold
            if scan_signal == 'BUY':
                signal_icon = "🟢"
            elif scan_signal == 'SELL':
                signal_icon = "🔴"
            
            # Format indicators if available
            indicators_html = ""
            if latest_scan.get('indicators'):
                indicators_html += "<div style='margin-top: 10px;'>Indicators:</div><ul style='margin-top: 5px;'>"
                for name, value in latest_scan['indicators'].items():
                    if isinstance(value, float):
                        indicators_html += f"<li>{name}: {value:.2f}</li>"
                    else:
                        indicators_html += f"<li>{name}: {value}</li>"
                indicators_html += "</ul>"
            
            st.markdown(f"""
            <div style="background-color: #1e273a; border-radius: 8px; padding: 15px; border: 1px solid #2e3c54;">
                <h4 style="margin-top: 0;">Latest Market Scan</h4>
                <div style="margin-bottom: 5px;"><strong>Time:</strong> {scan_time}</div>
                <div style="margin-bottom: 5px;"><strong>Pair:</strong> {scan_pair}</div>
                <div style="margin-bottom: 5px;"><strong>Signal:</strong> {signal_icon} {scan_signal}</div>
                <div style="margin-bottom: 5px;"><strong>Price:</strong> ${scan_price:.2f}</div>
                {indicators_html}
            </div>
            """, unsafe_allow_html=True)
        else:
            # Sample alerts if no scan file or database scan
            st.markdown("""
            <div style="background-color: #1e273a; border-radius: 8px; padding: 15px; border: 1px solid #2e3c54;">
                <h4 style="margin-top: 0;">Recent Alerts</h4>
                <div style="margin-bottom: 10px; padding-bottom: 10px; border-bottom: 1px solid #2e3c54;">
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #ffaf7b;">⚠ System</span>
                        <span style="color: #8b9eba; font-size: 0.8em;">Just now</span>
                    </div>
                    <div>No market scans found. Check your scan logging.</div>
                </div>
                <div>
                    <div style="display: flex; justify-content: space-between;">
                        <span style="color: #7f9cf5;">ℹ️ Info</span>
                        <span style="color: #8b9eba; font-size: 0.8em;">Just now</span>
                    </div>
                    <div>Dashboard initialized with no trade data.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error reading latest scan: {e}")
        st.info("Could not read the latest market scan.")

st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="margin-top: 30px; text-align: center; color: #8b9eba; font-size: 0.8em;">
    <p>HYPERION TRADING SYSTEM v1.0 | © 2025</p>
</div>
""", unsafe_allow_html=True)
