import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("dashboard")

# Check for database module
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
    initial_sidebar_state="expanded"
)

# Import CSS from external file (create a file named style.css in the same directory)
def load_css():
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    
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
try:
    load_css()
except:
    pass  # If style.css doesn't exist, we already loaded minimal inline CSS

# ----------------- DATA HANDLING FUNCTIONS -----------------

def generate_sample_data(days=30):
    """Generate sample trade data for testing purposes"""
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

def load_data_from_csv(csv_file=None):
    """Load trading data from CSV file with fallback to sample data"""
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

def load_data_from_database(start_date=None, end_date=None, pair=None, action=None):
    """Load trading data from the database"""
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

# ----------------- VISUALIZATION FUNCTIONS -----------------

def display_key_metrics(filtered_df):
    """Display key metrics at the top of the dashboard"""
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
        return

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div>TOTAL TRADES</div>
            <div class="metric-value metric-neutral">{total_trades}</div>
            <div>BUY: {buys} | SELL: {sells}</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        pnl_class = "metric-positive" if total_pnl >= 0 else "metric-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div>TOTAL PNL</div>
            <div class="metric-value {pnl_class}">${total_pnl:.2f}</div>
            <div>Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        win_count = len(profit_df[profit_df['net_profit'] > 0]) if not profit_df.empty else 0
        loss_count = len(profit_df[profit_df['net_profit'] < 0]) if not profit_df.empty else 0
        
        win_class = "metric-positive" if win_rate >= 50 else "metric-negative"
        st.markdown(f"""
        <div class="metric-card">
            <div>WIN RATE</div>
            <div class="metric-value {win_class}">{win_rate:.1f}%</div>
            <div>W: {win_count} | L: {loss_count}</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        # Calculate most profitable pair
        if not profit_df.empty:
            pair_profits = profit_df.groupby('pair')['net_profit'].sum()
            most_profitable_pair = pair_profits.idxmax() if not pair_profits.empty else "N/A"
            pair_profit = pair_profits.max() if not pair_profits.empty else 0
        else:
            most_profitable_pair = "N/A"
            pair_profit = 0
        
        st.markdown(f"""
        <div class="metric-card">
            <div>BEST PAIR</div>
            <div class="metric-value metric-positive">{most_profitable_pair}</div>
            <div>${pair_profit:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

def plot_cumulative_pnl(filtered_df):
    """Plot cumulative profit/loss over time"""
    try:
        if 'net_profit' in filtered_df.columns:
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
                    margin=dict(l=0, r=0, t=40, b=0)
                )
                
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("No sell trades found in the selected date range.")
        else:
            st.info("No profit data available.")
    except Exception as e:
        st.error(f"Error generating performance chart: {e}")

def display_recent_trades(filtered_df, limit=50):
    """Display a table of recent trades"""
    try:
        if not filtered_df.empty:
            # Sort by timestamp descending to show most recent first
            recent_trades = filtered_df.sort_values(by='timestamp', ascending=False).head(limit)
            
            # Create columns for display
            recent_trades['time'] = recent_trades['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
            recent_trades['price_formatted'] = recent_trades['price'].map('${:.2f}'.format)
            recent_trades['quantity_formatted'] = recent_trades['quantity'].map('{:.6f}'.format)
            
            # Format profit/loss
            def format_pnl(x):
                if pd.isna(x):
                    return "-"
                return f"${x:.2f}"
            
            recent_trades['pnl_formatted'] = recent_trades['net_profit'].apply(format_pnl)
            
            # Select and rename columns for display
            display_df = recent_trades[['time', 'pair', 'action', 'price_formatted', 'quantity_formatted', 'pnl_formatted']]
            display_df.columns = ['Time', 'Pair', 'Action', 'Price', 'Quantity', 'P/L']
            
            # Display the dataframe with custom formatting
            st.dataframe(
                display_df,
                use_container_width=True,
                height=400
            )
        else:
            st.info("No trades found for the selected filters.")
    except Exception as e:
        st.error(f"Error displaying recent trades: {e}")

def plot_pair_performance(filtered_df):
    """Plot performance by trading pair"""
    try:
        if 'net_profit' in filtered_df.columns and not filtered_df.empty:
            profit_df = filtered_df[filtered_df['net_profit'].notna()]
            
            if not profit_df.empty:
                # Calculate performance by pair
                pair_stats = profit_df.groupby('pair').agg({
                    'net_profit': 'sum',
                    'timestamp': 'count',
                }).rename(columns={'timestamp': 'trade_count'}).reset_index()
                
                # Calculate win rate by pair
                pair_win_rates = profit_df.groupby('pair')['net_profit'].apply(
                    lambda x: (x > 0).mean() * 100
                ).reset_index(name='win_rate')
                
                # Merge the stats
                pair_stats = pair_stats.merge(pair_win_rates, on='pair')
                pair_stats = pair_stats.sort_values('net_profit', ascending=False)
                
                # Create horizontal bar chart
                fig = px.bar(
                    pair_stats,
                    y='pair',
                    x='net_profit',
                    color='net_profit',
                    color_continuous_scale=['#ff5c87', '#7f9cf5', '#00ff9f'],
                    labels={'net_profit': 'Profit/Loss (USD)', 'pair': 'Trading Pair'},
                    orientation='h',
                    text='net_profit'
                )
                
                # Format the text to show values
                fig.update_traces(texttemplate='%{text:.2f}', textposition='outside')
                
                # Customize layout
                fig.update_layout(
                    title="Profit/Loss by Trading Pair",
                    template="plotly_dark",
                    paper_bgcolor='rgba(0,0,0,0)',
                    plot_bgcolor='rgba(0,0,0,0)',
                    coloraxis_showscale=False,
                    height=400
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Display the statistics in a table
                st.dataframe(
                    pair_stats.style.format({
                        'net_profit': '${:.2f}',
                        'win_rate': '{:.1f}%'
                    }),
                    use_container_width=True
                )
            else:
                st.info("No profit data available for the selected filters.")
        else:
            st.info("No profit data available for analysis.")
    except Exception as e:
        st.error(f"Error analyzing trading pairs: {e}")

# ----------------- MAIN APP -----------------

def main():
    # Create the header
    st.markdown('<h1 class="hyperion-header">HYPERION TRADING SYSTEM</h1>', unsafe_allow_html=True)
    
    # Current time and status
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.markdown(f'<p style="text-align: center;">LIVE SYSTEM · {current_time}</p>', unsafe_allow_html=True)
    
    # Add auto-refresh control in sidebar
    st.sidebar.markdown("### Auto Refresh")
    auto_refresh = st.sidebar.checkbox("Enable auto-refresh", value=False)
    if auto_refresh:
        refresh_rate = st.sidebar.slider("Refresh interval (seconds)", 5, 60, 30)
        st.write(f'<meta http-equiv="refresh" content="{refresh_rate}">', unsafe_allow_html=True)
    
    # Select data source
    st.sidebar.markdown("### Data Source")
    data_source = st.sidebar.radio(
        "Select data source:",
        ["Database", "CSV File", "Sample Data"],
        index=0 if HAS_DATABASE else 1
    )
    
    # Load data based on selected source
    if data_source == "Database":
        if HAS_DATABASE:
            # Add date range filter
            st.sidebar.markdown("### Date Range")
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
                # Switch to sample data
                df = generate_sample_data()
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
    
    # Filtering controls in sidebar
    st.sidebar.markdown("### Filters")
    
    # Pair filter
    try:
        pairs = ["All"] + sorted(df['pair'].unique().tolist())
    except:
        pairs = ["All", "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT"]
    
    selected_pair = st.sidebar.selectbox(
        "Trading Pair",
        options=pairs
    )
    
    # Action filter
    selected_action = st.sidebar.selectbox(
        "Action Type",
        options=["All", "BUY", "SELL"]
    )
    
    # Apply filters
    filtered_df = df.copy()
    
    # Pair filter
    if selected_pair != "All":
        filtered_df = filtered_df[filtered_df['pair'] == selected_pair]
    
    # Action filter
    if selected_action != "All":
        filtered_df = filtered_df[filtered_df['action'] == selected_action]
    
    # Date filter for CSV/Sample data (already applied for database)
    if data_source != "Database":
        # Add date filters to sidebar
        st.sidebar.markdown("### Date Range")
        try:
            min_date = df['timestamp'].min().date()
            max_date = df['timestamp'].max().date()
        except:
            min_date = (datetime.now() - timedelta(days=30)).date()
            max_date = datetime.now().date()
        
        start_date = st.sidebar.date_input(
            "Start Date",
            value=min_date,
            min_value=min_date,
            max_value=max_date
        )
        
        end_date = st.sidebar.date_input(
            "End Date",
            value=max_date,
            min_value=min_date,
            max_value=max_date
        )
        
        # Apply date filter
        filtered_df = filtered_df[
            (filtered_df['timestamp'].dt.date >= start_date) & 
            (filtered_df['timestamp'].dt.date <= end_date)
        ]
    
    # Bot controls in sidebar
    st.sidebar.markdown("### Bot Controls")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        if st.button("Start Bot"):
            st.sidebar.success("Bot started!")
    
    with col2:
        if st.button("Stop Bot"):
            st.sidebar.error("Bot stopped!")
    
    # Display key metrics
    display_key_metrics(filtered_df)
    
    # Create tabs for different sections
    tab1, tab2, tab3 = st.tabs(["Performance", "Trading Pairs", "Recent Trades"])
    
    with tab1:
        # Performance tab
        plot_cumulative_pnl(filtered_df)
    
    with tab2:
        # Trading pairs tab
        plot_pair_performance(filtered_df)
    
    with tab3:
        # Recent trades tab
        display_recent_trades(filtered_df)
    
    # Footer
    st.markdown("""
    <div style="margin-top: 30px; text-align: center; color: #8b9eba; font-size: 0.8em;">
        <p>HYPERION TRADING SYSTEM v1.0</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
