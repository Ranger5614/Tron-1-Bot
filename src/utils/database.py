"""
Database module for the trading bot dashboard.
This module provides a simplified interface to the main database functionality.
"""

import os
import sys
import pandas as pd
import logging
from datetime import datetime

# Add the root directory to the path to import the main database module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.utils.database import TradingDatabase, get_db

# Get logger
logger = logging.getLogger('crypto_bot.dashboard.database')

def get_trades(start_date=None, end_date=None, pair=None, action=None):
    """
    Get trades from the database with optional filtering.
    
    Args:
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        pair (str, optional): Trading pair to filter by
        action (str, optional): Action type to filter by (BUY/SELL)
        
    Returns:
        pandas.DataFrame: DataFrame containing the trades
    """
    try:
        # Get database instance
        db = get_db()
        
        # Get trades from database
        df = db.get_trades(
            pair=pair,
            action=action,
            start_date=start_date,
            end_date=end_date
        )
        
        # Convert timestamp to datetime if it's a string
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Calculate cumulative profit if not present
        if 'cumulative_net_profit' not in df.columns and 'net_profit' in df.columns:
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
        
        return df
    except Exception as e:
        logger.error(f"Error getting trades from database: {e}")
        return pd.DataFrame()

def get_latest_status():
    """
    Get the latest bot status from the database.
    
    Returns:
        dict: Dictionary containing the latest bot status
    """
    try:
        db = get_db()
        return db.get_latest_status()
    except Exception as e:
        logger.error(f"Error getting latest bot status: {e}")
        return {}

def import_from_csv(csv_file):
    """
    Import trades from a CSV file into the database.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db = get_db()
        db.import_from_csv(csv_file)
        return True
    except Exception as e:
        logger.error(f"Error importing from CSV: {e}")
        return False

def export_to_csv(csv_file, start_date=None, end_date=None):
    """
    Export trades from the database to a CSV file.
    
    Args:
        csv_file (str): Path to the CSV file
        start_date (str, optional): Start date in YYYY-MM-DD format
        end_date (str, optional): End date in YYYY-MM-DD format
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        db = get_db()
        db.export_to_csv(csv_file, start_date, end_date)
        return True
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return False 