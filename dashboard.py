
import streamlit as st
import pandas as pd
import os

st.set_page_config(page_title="TRON 1 Dashboard", layout="wide")

st.title("🚀 TRON 1 Live Trading Dashboard")

# Load data
csv_file = "trade_log.csv"

if not os.path.exists(csv_file):
    st.warning("No trade_log.csv found. Make a trade first!")
else:
    df = pd.read_csv(csv_file)

    # Convert timestamp column to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Sidebar filters
    st.sidebar.header("🔍 Filters")
    selected_pair = st.sidebar.selectbox("Select Pair", options=["All"] + sorted(df['pair'].unique().tolist()))

    if selected_pair != "All":
        df = df[df['pair'] == selected_pair]

    # Metrics
    total_trades = len(df)
    total_pnl = df['pnl'].dropna().sum()
    win_rate = (df['pnl'].dropna() > 0).mean() * 100 if not df['pnl'].dropna().empty else 0

    col1, col2, col3 = st.columns(3)
    col1.metric("🔁 Total Trades", total_trades)
    col2.metric("💰 Total PnL", f"${total_pnl:.2f}")
    col3.metric("🏆 Win Rate", f"{win_rate:.1f}%")

    # Charts
    st.subheader("📈 PnL Over Time")
    df['cumulative_pnl'] = df['pnl'].fillna(0).cumsum()
    st.line_chart(df.set_index('timestamp')['cumulative_pnl'])

    st.subheader("📊 PnL by Pair")
    pnl_by_pair = df.groupby('pair')['pnl'].sum().sort_values(ascending=False)
    st.bar_chart(pnl_by_pair)

    # Recent trades
    st.subheader("📋 Recent Trades")
    st.dataframe(df.sort_values(by='timestamp', ascending=False).reset_index(drop=True))
