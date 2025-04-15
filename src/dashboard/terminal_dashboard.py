#!/usr/bin/env python
"""
HYPERION Trading System - Terminal Dashboard
A sci-fi themed terminal interface for monitoring trading operations.
"""

import os
import sys
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.live import Live
from rich.text import Text
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.style import Style
from rich import box
import pandas as pd

# Add the root directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.database import get_trades, get_latest_status

# Initialize Rich console
console = Console()

# Sci-fi color scheme
HYPERION_BLUE = Style(color="cyan")
HYPERION_GREEN = Style(color="green")
HYPERION_RED = Style(color="red")
HYPERION_YELLOW = Style(color="yellow")
HYPERION_PURPLE = Style(color="magenta")
HYPERION_WHITE = Style(color="white")

class HyperionDashboard:
    def __init__(self):
        self.console = Console()
        self.layout = Layout()
        self.layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=3)
        )
        self.layout["main"].split_row(
            Layout(name="left"),
            Layout(name="right")
        )
        self.layout["left"].split_column(
            Layout(name="metrics"),
            Layout(name="trades")
        )
        self.layout["right"].split_column(
            Layout(name="status"),
            Layout(name="performance")
        )

    def generate_header(self):
        """Generate the sci-fi themed header"""
        title = Text("HYPERION TRADING SYSTEM", style=HYPERION_BLUE)
        timestamp = Text(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", style=HYPERION_WHITE)
        return Panel(
            title + "\n" + timestamp,
            style=HYPERION_BLUE,
            box=box.DOUBLE
        )

    def generate_metrics(self, trades_df):
        """Generate trading metrics panel"""
        if trades_df.empty:
            return Panel("No trading data available", style=HYPERION_YELLOW)

        total_trades = len(trades_df)
        total_pnl = trades_df['net_profit'].sum() if 'net_profit' in trades_df.columns else 0
        win_rate = (trades_df['net_profit'] > 0).mean() * 100 if 'net_profit' in trades_df.columns else 0

        metrics_text = Text()
        metrics_text.append(f"Total Trades: {total_trades}\n", style=HYPERION_WHITE)
        metrics_text.append(f"Total PnL: ${total_pnl:.2f}\n", 
                          style=HYPERION_GREEN if total_pnl >= 0 else HYPERION_RED)
        metrics_text.append(f"Win Rate: {win_rate:.1f}%\n", style=HYPERION_WHITE)

        return Panel(metrics_text, title="Trading Metrics", style=HYPERION_BLUE, box=box.DOUBLE)

    def generate_trades_table(self, trades_df):
        """Generate recent trades table"""
        if trades_df.empty:
            return Panel("No trades available", style=HYPERION_YELLOW)

        table = Table(show_header=True, header_style=HYPERION_BLUE, box=box.DOUBLE)
        table.add_column("Time", style=HYPERION_WHITE)
        table.add_column("Pair", style=HYPERION_WHITE)
        table.add_column("Action", style=HYPERION_WHITE)
        table.add_column("PnL", style=HYPERION_WHITE)

        # Get last 5 trades
        recent_trades = trades_df.tail(5)
        for _, trade in recent_trades.iterrows():
            pnl = trade.get('net_profit', 0)
            pnl_style = HYPERION_GREEN if pnl >= 0 else HYPERION_RED
            table.add_row(
                str(trade.get('timestamp', '')),
                str(trade.get('pair', '')),
                str(trade.get('action', '')),
                Text(f"${pnl:.2f}", style=pnl_style)
            )

        return Panel(table, title="Recent Trades", style=HYPERION_BLUE, box=box.DOUBLE)

    def generate_status(self, status):
        """Generate bot status panel"""
        status_text = Text()
        status_text.append("Bot Status: ", style=HYPERION_WHITE)
        status_text.append(status.get('status', 'UNKNOWN'), 
                         style=HYPERION_GREEN if status.get('status') == 'RUNNING' else HYPERION_RED)
        status_text.append(f"\nAccount Value: ${status.get('account_value', 0):.2f}", style=HYPERION_WHITE)
        
        return Panel(status_text, title="System Status", style=HYPERION_BLUE, box=box.DOUBLE)

    def generate_performance(self, trades_df):
        """Generate performance visualization"""
        if trades_df.empty:
            return Panel("No performance data available", style=HYPERION_YELLOW)

        # Simple ASCII chart for cumulative PnL
        if 'cumulative_net_profit' in trades_df.columns:
            pnl = trades_df['cumulative_net_profit'].iloc[-1]
            chart = "█" * min(abs(int(pnl/100)), 50)  # Simple bar chart
            chart_text = Text()
            chart_text.append(f"Cumulative PnL: ${pnl:.2f}\n", 
                            style=HYPERION_GREEN if pnl >= 0 else HYPERION_RED)
            chart_text.append(chart, style=HYPERION_GREEN if pnl >= 0 else HYPERION_RED)
            return Panel(chart_text, title="Performance", style=HYPERION_BLUE, box=box.DOUBLE)
        
        return Panel("No performance data available", style=HYPERION_YELLOW)

    def generate_footer(self):
        """Generate the footer with system information"""
        footer_text = Text()
        footer_text.append("HYPERION TRADING SYSTEM v1.0", style=HYPERION_PURPLE)
        footer_text.append(" | ")
        footer_text.append("Press Ctrl+C to exit", style=HYPERION_WHITE)
        return Panel(footer_text, style=HYPERION_BLUE, box=box.DOUBLE)

    def update(self):
        """Update the dashboard with latest data"""
        try:
            # Get latest data
            trades_df = get_trades()
            status = get_latest_status()

            # Update layout
            self.layout["header"].update(self.generate_header())
            self.layout["metrics"].update(self.generate_metrics(trades_df))
            self.layout["trades"].update(self.generate_trades_table(trades_df))
            self.layout["status"].update(self.generate_status(status))
            self.layout["performance"].update(self.generate_performance(trades_df))
            self.layout["footer"].update(self.generate_footer())

            return self.layout
        except Exception as e:
            return Panel(f"Error updating dashboard: {str(e)}", style=HYPERION_RED)

def main():
    """Main function to run the terminal dashboard"""
    dashboard = HyperionDashboard()
    
    with Live(dashboard.layout, refresh_per_second=1) as live:
        try:
            while True:
                live.update(dashboard.update())
                time.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[bold red]Shutting down HYPERION Trading System...[/bold red]")
            sys.exit(0)

if __name__ == "__main__":
    main() 