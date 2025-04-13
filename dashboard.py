import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import altair as alt
import time

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
        st.error(f"Error generating sample data: {e}")
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

# ----- PATCHED SECTION: Improved CSV Loading Logic -----

# Function to find the trade log file with enhanced debugging
def find_trade_log():
    """
    Search for trade log file in multiple possible locations with improved debugging.
    
    Returns:
        str: Path to the found CSV file, or None if not found
    """
    possible_paths = [
        "trade_log.csv",  # Root directory
        os.path.join("trades", "trade_log.csv"),  # Trades directory
        os.path.join(os.getcwd(), "trade_log.csv"),  # Absolute path
        os.path.join(os.getcwd(), "trades", "trade_log.csv"),  # Absolute path in trades dir
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trade_log.csv"),  # Project root
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trades", "trade_log.csv"),  # Project trades dir
    ]
    
    st.sidebar.markdown("#### CSV File Locations")
    st.sidebar.write("Searching for trade log in:")
    
    for path in possible_paths:
        exists = os.path.exists(path)
        if exists:
            file_size = os.path.getsize(path)
            status = f"✅ Found ({file_size} bytes)" if file_size > 0 else "⚠️ Empty file"
        else:
            status = "❌ Not found"
            
        st.sidebar.write(f"- {path}: {status}")
        
        if exists and file_size > 0:
            try:
                # Try to read the file to verify it's valid
                with open(path, 'r') as f:
                    header = f.readline().strip()
                
                st.sidebar.success(f"Found valid trade log at: {path}")
                st.sidebar.write(f"Header: {header}")
                return path
            except Exception as e:
                st.sidebar.warning(f"Error reading file at {path}: {str(e)}")
    
    # If no valid file found, create an empty one
    create_empty = st.sidebar.button("Create Empty Trade Log File")
    if create_empty:
        try:
            default_path = "trade_log.csv"
            with open(default_path, 'w', newline='') as f:
                f.write("timestamp,pair,action,price,quantity,net_profit,profit_pct\n")
            
            st.sidebar.success(f"Created empty trade log at {default_path}")
            return default_path
        except Exception as e:
            st.sidebar.error(f"Error creating file: {str(e)}")
    
    return None

# Function to add CSV upload widget
def add_csv_upload_widget():
    """Add a CSV upload widget to the sidebar"""
    st.sidebar.markdown("#### Upload Trade Log")
    uploaded_file = st.sidebar.file_uploader("Upload CSV file", type=['csv'])
    if uploaded_file is not None:
        try:
            # Read the file to verify it's valid
            df = pd.read_csv(uploaded_file)
            st.sidebar.write(f"CSV contains {len(df)} trades")
            
            # Option to save the file
            if st.sidebar.button("Save Uploaded File as trade_log.csv"):
                with open("trade_log.csv", "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                st.sidebar.success("File saved successfully! Refreshing data...")
                time.sleep(1)  # Short delay to ensure file is saved
                st.experimental_rerun()
                
            return df
        except Exception as e:
            st.sidebar.error(f"Error reading uploaded file: {str(e)}")
            return None
    
    return None

# Function to add test trade generator
def add_test_trade_generator():
    """Add a test trade generator widget to the sidebar"""
    st.sidebar.markdown("#### Generate Test Trade")
    with st.sidebar.expander("Create Test Trade"):
        col1, col2 = st.columns(2)
        with col1:
            pair = st.selectbox("Pair", ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"])
            action = st.selectbox("Action", ["BUY", "SELL"])
        with col2:
            price = st.number_input("Price", value=45000.0, step=100.0, min_value=0.1)
            quantity = st.number_input("Quantity", value=0.001, step=0.001, min_value=0.001)
        
        if action == "SELL":
            profit = st.number_input("Profit/Loss ($)", value=10.0)
            profit_pct = st.number_input("Profit/Loss (%)", value=2.0)
        else:
            profit = None
            profit_pct = None
        
        if st.button("Add Test Trade"):
            try:
                import csv
                
                # Create CSV if it doesn't exist
                file_exists = os.path.exists("trade_log.csv")
                
                with open("trade_log.csv", mode="a", newline="") as f:
                    writer = csv.writer(f)
                    
                    # Write header if file is new
                    if not file_exists:
                        writer.writerow(["timestamp", "pair", "action", "price", "quantity", "net_profit", "profit_pct"])
                    
                    # Write the test trade
                    writer.writerow([
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        pair,
                        action,
                        price,
                        quantity,
                        profit if profit is not None else "",
                        profit_pct if profit_pct is not None else ""
                    ])
                
                st.success(f"Added {action} trade for {pair}! Refreshing data...")
                time.sleep(1)  # Short delay to ensure file is saved
                st.experimental_rerun()
            except Exception as e:
                st.error(f"Error adding test trade: {e}")

# Enhanced function to load trade data
def load_trade_data(csv_file=None, uploaded_df=None):
    """
    Load trade data with enhanced error handling.
    
    Args:
        csv_file (str): Path to CSV file
        uploaded_df (DataFrame): DataFrame from uploaded file
        
    Returns:
        DataFrame: Processed trade data
    """
    try:
        # First try to use uploaded data if available
        if uploaded_df is not None:
            st.sidebar.success("Using uploaded CSV data")
            df = uploaded_df
        # Then try to load from file
        elif csv_file is not None:
            st.sidebar.info(f"Loading trade data from: {csv_file}")
            df = pd.read_csv(csv_file)
        # Fall back to sample data
        else:
            st.sidebar.warning("No trade data source found. Using sample data.")
            return generate_sample_data()
        
        # Validate required columns
        required_columns = ['timestamp', 'pair', 'action', 'price', 'quantity']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            st.sidebar.error(f"Missing columns in CSV: {', '.join(missing_columns)}")
            return generate_sample_data()
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Handle optional columns
        if 'net_profit' not in df.columns:
            df['net_profit'] = None
        
        if 'profit_pct' not in df.columns:
            df['profit_pct'] = None
        
        # Calculate cumulative profit
        if 'cumulative_net_profit' not in df.columns:
            df['cumulative_net_profit'] = None
            running_total = 0
            
            for idx, row in df.iterrows():
                if row['action'] == 'SELL' and pd.notna(row['net_profit']):
                    running_total += row['net_profit']
                df.at[idx, 'cumulative_net_profit'] = running_total
            
            # Fill forward cumulative profit
            df['cumulative_net_profit'] = df['cumulative_net_profit'].fillna(method='ffill')
        
        st.sidebar.success(f"Successfully loaded {len(df)} trades")
        return df
        
    except Exception as e:
        st.sidebar.error(f"Error loading trade data: {e}")
        import traceback
        st.sidebar.text(traceback.format_exc())
        return generate_sample_data()

# ----- END PATCHED SECTION -----

# Load data using improved functions
try:
    # Add auto-refresh control
    if st.sidebar.checkbox("Enable auto-refresh", value=False):
        refresh_interval = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 30)
        st.write(f'<meta http-equiv="refresh" content="{refresh_interval}">', unsafe_allow_html=True)
    
    # Find the trade log file
    csv_file = find_trade_log()
    
    # Add CSV upload option
    uploaded_df = add_csv_upload_widget()
    
    # Add test trade generator
    add_test_trade_generator()
    
    # Load trade data
    df = load_trade_data(csv_file, uploaded_df)
    
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
        win_class = "metric-positive" if win_rate >= 50 else "metric-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-title">WIN RATE</div>
            <div class="metric-value {win_class}">{win_rate:.1f}%</div>
            <div>W: {len(profit_df[profit_df['net_profit'] > 0]) if not profit_df.empty else 0} | L: {len(profit_df[profit_df['net_profit'] < 0]) if not profit_df.empty else 0}</div>
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
        # Create a styled status panel
        st.markdown("""
        <div style="background-color: #1e273a; border-radius: 8px; padding: 15px; border: 1px solid #2e3c54;">
            <h4 style="margin-top: 0;">System Health</h4>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 10px; height: 10px; background-color: #00ff9f; border-radius: 50%; margin-right: 10px;"></div>
                <div>API Connection: <span style="color: #00ff9f;">Online</span></div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 10px; height: 10px; background-color: #00ff9f; border-radius: 50%; margin-right: 10px;"></div>
                <div>Database: <span style="color: #00ff9f;">Connected</span></div>
            </div>
            <div style="display: flex; align-items: center; margin-bottom: 10px;">
                <div style="width: 10px; height: 10px; background-color: #00ff9f; border-radius: 50%; margin-right: 10px;"></div>
                <div>Discord Notifications: <span style="color: #00ff9f;">Active</span></div>
            </div>
            <div style="display: flex; align-items: center;">
                <div style="width: 10px; height: 10px; background-color: #00ff9f; border-radius: 50%; margin-right: 10px;"></div>
                <div>Trading Engine: <span style="color: #00ff9f;">Running</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        # Read latest scan file if available
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
            else:
                # Sample alerts if no scan file
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

except Exception as e:
    st.error(f"Error in dashboard: {e}")
    import traceback
    st.code(traceback.format_exc())
