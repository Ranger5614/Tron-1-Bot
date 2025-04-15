import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import sys
import logging

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

# Set Streamlit page configuration
st.set_page_config(
    page_title="HYPERION Trading System",
    page_icon="🔱",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Import CSS from external file
def load_css():
    try:
        # Try to load from the dashboard directory first
        css_path = os.path.join(os.path.dirname(__file__), "style.css")
        if os.path.exists(css_path):
            with open(css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                return
        
        # Try to load from the root directory
        root_css_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "style.css")
        if os.path.exists(root_css_path):
            with open(root_css_path) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
                return
    except Exception as e:
        logger.warning(f"Error loading CSS: {e}")
    
    # In case file doesn't exist, load minimal inline CSS
    st.markdown("""
    <style>
        .main { background-color: #0e1117; color: #e0e0e0; }
        .hyperion-header { 
            background: linear-gradient(90deg, #3a1c71, #d76d77, #ffaf7b);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3em;
            text-align: center;
        }
        .metric-card {
            background-color: #1e2430;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            margin: 10px 0;
        }
        .metric-positive { color: #00ff9f; }
        .metric-negative { color: #ff5c87; }
        .metric-neutral { color: #e0e0e0; }
    </style>
    <link href="https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700&display=swap" rel="stylesheet">
    """, unsafe_allow_html=True)

# Load CSS
load_css()

# ----------------- DATA HANDLING FUNCTIONS -----------------

def load_data_from_csv(csv_file=None):
    """Load trading data from CSV file"""
    try:
        # Default to trade_log.csv if no file specified
        if csv_file is None:
            csv_file = "trade_log.csv"
            
        if not os.path.exists(csv_file):
            logger.warning(f"CSV file not found: {csv_file}")
            return pd.DataFrame()
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate cumulative profit
        df['cumulative_net_profit'] = df['pnl'].cumsum()
        
        logger.info(f"Loaded {len(df)} trades from CSV: {csv_file}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from CSV: {e}")
        return pd.DataFrame()

def display_key_metrics(df, strategy_type):
    """Display key trading metrics for a specific strategy"""
    if df.empty:
        st.warning(f"No data available for {strategy_type} strategy")
        return
    
    # Filter data based on strategy type
    if strategy_type == "Short Term":
        filtered_df = df[df['pair'].isin(['BTCUSDT', 'ETHUSDT'])]  # Example filter
    elif strategy_type == "Medium Term":
        filtered_df = df[df['pair'].isin(['SOLUSDT', 'XRPUSDT'])]  # Example filter
    else:  # Long Term
        filtered_df = df[df['pair'].isin(['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT'])]
    
    # Calculate metrics
    total_trades = len(filtered_df)
    winning_trades = len(filtered_df[filtered_df['pnl'] > 0])
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = filtered_df['pnl'].sum()
    avg_pnl = filtered_df['pnl'].mean() if total_trades > 0 else 0
    
    # Display metrics in columns
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric("Total Trades", total_trades)
    with col2:
        st.metric("Win Rate", f"{win_rate:.1f}%")
    with col3:
        st.metric("Total PnL", f"${total_pnl:.2f}", delta=f"${total_pnl:.2f}")
    with col4:
        st.metric("Avg PnL", f"${avg_pnl:.2f}")
    with col5:
        st.metric("Active Pairs", len(filtered_df['pair'].unique()))

def plot_strategy_performance(df, strategy_type):
    """Plot performance metrics for a specific strategy"""
    if df.empty:
        return
    
    # Filter data based on strategy type
    if strategy_type == "Short Term":
        filtered_df = df[df['pair'].isin(['BTCUSDT', 'ETHUSDT'])]
    elif strategy_type == "Medium Term":
        filtered_df = df[df['pair'].isin(['SOLUSDT', 'XRPUSDT'])]
    else:  # Long Term
        filtered_df = df
    
    # Create cumulative PnL plot
    fig = go.Figure()
    
    # Add cumulative PnL line
    fig.add_trace(go.Scatter(
        x=filtered_df['timestamp'],
        y=filtered_df['cumulative_net_profit'],
        mode='lines',
        name='Cumulative PnL',
        line=dict(color='#00ff9f')
    ))
    
    # Add buy/sell markers
    buys = filtered_df[filtered_df['action'] == 'BUY']
    sells = filtered_df[filtered_df['action'] == 'SELL']
    
    fig.add_trace(go.Scatter(
        x=buys['timestamp'],
        y=buys['cumulative_net_profit'],
        mode='markers',
        name='Buy',
        marker=dict(color='#00ff9f', size=10, symbol='triangle-up')
    ))
    
    fig.add_trace(go.Scatter(
        x=sells['timestamp'],
        y=sells['cumulative_net_profit'],
        mode='markers',
        name='Sell',
        marker=dict(color='#ff5c87', size=10, symbol='triangle-down')
    ))
    
    # Update layout
    fig.update_layout(
        title=f"{strategy_type} Strategy Performance",
        xaxis_title="Time",
        yaxis_title="Cumulative PnL ($)",
        template="plotly_dark",
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)

def display_recent_trades(df, strategy_type):
    """Display recent trades for a specific strategy"""
    if df.empty:
        return
    
    # Filter data based on strategy type
    if strategy_type == "Short Term":
        filtered_df = df[df['pair'].isin(['BTCUSDT', 'ETHUSDT'])]
    elif strategy_type == "Medium Term":
        filtered_df = df[df['pair'].isin(['SOLUSDT', 'XRPUSDT'])]
    else:  # Long Term
        filtered_df = df
    
    # Get recent trades
    recent_trades = filtered_df.sort_values('timestamp', ascending=False).head(10)
    
    # Display trades in a table
    st.subheader("Recent Trades")
    st.dataframe(
        recent_trades[['timestamp', 'pair', 'action', 'price', 'quantity', 'pnl', 'pnl_pct']],
        use_container_width=True
    )

def main():
    # Create the header
    st.markdown('<h1 class="hyperion-header">HYPERION Trading System</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data_from_csv()
    
    if df.empty:
        st.error("No trading data available. Please check your trade_log.csv file.")
        return
    
    # Create tabs for each strategy
    st.markdown("## Strategy Performance")
    
    # Short Term Strategy
    st.markdown("### Short Term Strategy (Scalping)")
    display_key_metrics(df, "Short Term")
    plot_strategy_performance(df, "Short Term")
    display_recent_trades(df, "Short Term")
    
    # Medium Term Strategy
    st.markdown("### Medium Term Strategy (TRON 1.1)")
    display_key_metrics(df, "Medium Term")
    plot_strategy_performance(df, "Medium Term")
    display_recent_trades(df, "Medium Term")
    
    # Long Term Strategy
    st.markdown("### Long Term Strategy")
    display_key_metrics(df, "Long Term")
    plot_strategy_performance(df, "Long Term")
    display_recent_trades(df, "Long Term")

if __name__ == "__main__":
    main() 