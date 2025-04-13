"""
database.py - SQL database module for the trading bot
"""

import os
import sqlite3
import pandas as pd
import logging
from datetime import datetime

# Get logger
logger = logging.getLogger('crypto_bot.database')

class TradingDatabase:
    """
    Database handler for storing and retrieving trading data.
    Uses SQLite for local storage.
    """
    
    def __init__(self, db_file=None):
        """
        Initialize the database connection.
        
        Args:
            db_file (str, optional): Path to the SQLite database file.
                If None, 'trading_bot.db' in the current directory is used.
        """
        # Use default path if none provided
        if db_file is None:
            # Create in current directory by default
            db_file = os.path.join(os.getcwd(), "trading_bot.db")
        
        self.db_file = db_file
        self.conn = None
        
        # Initialize connection and tables
        try:
            self.connect()
            self.create_tables()
            logger.info(f"Database initialized at {self.db_file}")
        except Exception as e:
            logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def connect(self):
        """Connect to the SQLite database."""
        try:
            self.conn = sqlite3.connect(self.db_file)
            # Enable foreign keys
            self.conn.execute("PRAGMA foreign_keys = ON")
            logger.info(f"Connected to database: {self.db_file}")
        except Exception as e:
            logger.error(f"Database connection error: {str(e)}")
            raise
    
    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None
            logger.info("Database connection closed")
    
    def create_tables(self):
        """Create necessary database tables if they don't exist."""
        try:
            cursor = self.conn.cursor()
            
            # Create trades table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                action TEXT NOT NULL,
                price REAL NOT NULL,
                quantity REAL NOT NULL,
                net_profit REAL,
                profit_pct REAL,
                order_id TEXT,
                strategy TEXT,
                created_at TEXT NOT NULL
            )
            ''')
            
            # Create market_scans table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_scans (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                pair TEXT NOT NULL,
                signal TEXT,
                price REAL,
                strategy TEXT,
                interval TEXT,
                indicators TEXT,
                created_at TEXT NOT NULL
            )
            ''')
            
            # Create bot_status table
            cursor.execute('''
            CREATE TABLE IF NOT EXISTS bot_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                status TEXT NOT NULL,
                account_value REAL,
                active_pairs TEXT,
                message TEXT,
                created_at TEXT NOT NULL
            )
            ''')
            
            # Create index on timestamp for faster queries
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_pair ON trades(pair)')
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_market_scans_timestamp ON market_scans(timestamp)')
            
            self.conn.commit()
            logger.info("Database tables created successfully")
        except Exception as e:
            logger.error(f"Error creating database tables: {str(e)}")
            raise
    
    def log_trade(self, pair, action, price, quantity, net_profit=None, profit_pct=None, order_id=None, strategy=None):
        """
        Log a trade to the database.
        
        Args:
            pair (str): Trading pair (e.g., 'BTCUSDT')
            action (str): Trade action ('BUY' or 'SELL')
            price (float): Trade price
            quantity (float): Trade quantity
            net_profit (float, optional): Profit/Loss in currency. Defaults to None.
            profit_pct (float, optional): Profit/Loss percentage. Defaults to None.
            order_id (str, optional): Exchange order ID. Defaults to None.
            strategy (str, optional): Strategy used for this trade. Defaults to None.
            
        Returns:
            int: The ID of the inserted trade
        """
        try:
            # Ensure connection is active
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
            cursor.execute('''
            INSERT INTO trades (timestamp, pair, action, price, quantity, net_profit, profit_pct, order_id, strategy, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                pair,
                action,
                price,
                quantity,
                net_profit,
                profit_pct,
                order_id,
                strategy,
                timestamp  # created_at is the same as timestamp for new entries
            ))
            
            self.conn.commit()
            trade_id = cursor.lastrowid
            logger.info(f"Trade logged to database: {pair} {action} @ {price}, ID: {trade_id}")
            
            return trade_id
        except Exception as e:
            logger.error(f"Error logging trade to database: {str(e)}")
            if self.conn:
                self.conn.rollback()
            raise
    
    def log_market_scan(self, pair, signal, price, strategy=None, interval=None, indicators=None):
        """
        Log a market scan to the database.
        
        Args:
            pair (str): Trading pair (e.g., 'BTCUSDT')
            signal (str): Signal detected (e.g., 'BUY', 'SELL', 'HOLD')
            price (float): Current price
            strategy (str, optional): Strategy used. Defaults to None.
            interval (str, optional): Timeframe interval. Defaults to None.
            indicators (dict, optional): Dictionary of indicator values. Defaults to None.
            
        Returns:
            int: The ID of the inserted scan
        """
        try:
            # Ensure connection is active
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert indicators dict to string if provided
            indicators_str = None
            if indicators:
                import json
                indicators_str = json.dumps(indicators)
            
            cursor.execute('''
            INSERT INTO market_scans (timestamp, pair, signal, price, strategy, interval, indicators, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                pair,
                signal,
                price,
                strategy,
                interval,
                indicators_str,
                timestamp  # created_at is the same as timestamp for new entries
            ))
            
            self.conn.commit()
            scan_id = cursor.lastrowid
            logger.info(f"Market scan logged to database: {pair} {signal} @ {price}, ID: {scan_id}")
            
            return scan_id
        except Exception as e:
            logger.error(f"Error logging market scan to database: {str(e)}")
            if self.conn:
                self.conn.rollback()
            raise
    
    def log_bot_status(self, status, account_value=None, active_pairs=None, message=None):
        """
        Log the bot's status to the database.
        
        Args:
            status (str): Bot status (e.g., 'RUNNING', 'STOPPED', 'ERROR')
            account_value (float, optional): Current account value. Defaults to None.
            active_pairs (list, optional): List of active trading pairs. Defaults to None.
            message (str, optional): Optional status message. Defaults to None.
            
        Returns:
            int: The ID of the inserted status entry
        """
        try:
            # Ensure connection is active
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert active_pairs list to string if provided
            active_pairs_str = None
            if active_pairs:
                if isinstance(active_pairs, list):
                    active_pairs_str = ','.join(active_pairs)
                else:
                    active_pairs_str = str(active_pairs)
            
            cursor.execute('''
            INSERT INTO bot_status (timestamp, status, account_value, active_pairs, message, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                timestamp,
                status,
                account_value,
                active_pairs_str,
                message,
                timestamp  # created_at is the same as timestamp for new entries
            ))
            
            self.conn.commit()
            status_id = cursor.lastrowid
            logger.info(f"Bot status logged to database: {status}, ID: {status_id}")
            
            return status_id
        except Exception as e:
            logger.error(f"Error logging bot status to database: {str(e)}")
            if self.conn:
                self.conn.rollback()
            raise
    
    def get_trades(self, pair=None, action=None, start_date=None, end_date=None, limit=None):
        """
        Get trades from the database with optional filtering.
        
        Args:
            pair (str, optional): Filter by trading pair. Defaults to None.
            action (str, optional): Filter by action ('BUY' or 'SELL'). Defaults to None.
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to None.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to None.
            
        Returns:
            pandas.DataFrame: DataFrame containing the trades
        """
        try:
            # Ensure connection is active
            if not self.conn:
                self.connect()
            
            # Build query
            query = "SELECT * FROM trades WHERE 1=1"
            params = []
            
            if pair:
                query += " AND pair = ?"
                params.append(pair)
            
            if action:
                query += " AND action = ?"
                params.append(action)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(f"{start_date} 00:00:00")
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(f"{end_date} 23:59:59")
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            # Execute query and return as DataFrame
            df = pd.read_sql_query(query, self.conn, params=params)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Calculate cumulative profit
            if not df.empty and 'net_profit' in df.columns:
                # Sort by timestamp (oldest first) for proper cumulative calculation
                df_sorted = df.sort_values('timestamp')
                
                # Calculate cumulative profit
                df_sorted['cumulative_net_profit'] = df_sorted['net_profit'].fillna(0).cumsum()
                
                # Resort to original order
                df = df_sorted.sort_values('timestamp', ascending=False).reset_index(drop=True)
            
            logger.info(f"Retrieved {len(df)} trades from database")
            return df
        except Exception as e:
            logger.error(f"Error getting trades from database: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def get_market_scans(self, pair=None, signal=None, start_date=None, end_date=None, limit=None):
        """
        Get market scans from the database with optional filtering.
        
        Args:
            pair (str, optional): Filter by trading pair. Defaults to None.
            signal (str, optional): Filter by signal. Defaults to None.
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to None.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to None.
            
        Returns:
            pandas.DataFrame: DataFrame containing the market scans
        """
        try:
            # Ensure connection is active
            if not self.conn:
                self.connect()
            
            # Build query
            query = "SELECT * FROM market_scans WHERE 1=1"
            params = []
            
            if pair:
                query += " AND pair = ?"
                params.append(pair)
            
            if signal:
                query += " AND signal = ?"
                params.append(signal)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(f"{start_date} 00:00:00")
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(f"{end_date} 23:59:59")
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            # Execute query and return as DataFrame
            df = pd.read_sql_query(query, self.conn, params=params)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Parse indicators json
            if not df.empty and 'indicators' in df.columns:
                import json
                
                def parse_indicators(indicators_str):
                    if pd.isna(indicators_str) or not indicators_str:
                        return {}
                    try:
                        return json.loads(indicators_str)
                    except:
                        return {}
                
                df['indicators_dict'] = df['indicators'].apply(parse_indicators)
            
            logger.info(f"Retrieved {len(df)} market scans from database")
            return df
        except Exception as e:
            logger.error(f"Error getting market scans from database: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def get_bot_status(self, status=None, start_date=None, end_date=None, limit=None):
        """
        Get bot status entries from the database with optional filtering.
        
        Args:
            status (str, optional): Filter by status. Defaults to None.
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to None.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to None.
            limit (int, optional): Limit the number of results. Defaults to None.
            
        Returns:
            pandas.DataFrame: DataFrame containing the bot status entries
        """
        try:
            # Ensure connection is active
            if not self.conn:
                self.connect()
            
            # Build query
            query = "SELECT * FROM bot_status WHERE 1=1"
            params = []
            
            if status:
                query += " AND status = ?"
                params.append(status)
            
            if start_date:
                query += " AND timestamp >= ?"
                params.append(f"{start_date} 00:00:00")
            
            if end_date:
                query += " AND timestamp <= ?"
                params.append(f"{end_date} 23:59:59")
            
            query += " ORDER BY timestamp DESC"
            
            if limit:
                query += " LIMIT ?"
                params.append(limit)
            
            # Execute query and return as DataFrame
            df = pd.read_sql_query(query, self.conn, params=params)
            
            # Convert timestamp to datetime
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Parse active_pairs
            if not df.empty and 'active_pairs' in df.columns:
                def parse_active_pairs(pairs_str):
                    if pd.isna(pairs_str) or not pairs_str:
                        return []
                    return pairs_str.split(',')
                
                df['active_pairs_list'] = df['active_pairs'].apply(parse_active_pairs)
            
            logger.info(f"Retrieved {len(df)} bot status entries from database")
            return df
        except Exception as e:
            logger.error(f"Error getting bot status from database: {str(e)}")
            return pd.DataFrame()  # Return empty DataFrame on error
    
    def get_latest_scan(self, pair=None):
        """
        Get the most recent market scan for a pair.
        
        Args:
            pair (str, optional): Trading pair. If None, gets latest scan for any pair.
            
        Returns:
            dict: The latest scan information, or None if not found
        """
        try:
            # Ensure connection is active
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            
            query = "SELECT * FROM market_scans"
            params = []
            
            if pair:
                query += " WHERE pair = ?"
                params.append(pair)
            
            query += " ORDER BY timestamp DESC LIMIT 1"
            
            cursor.execute(query, params)
            result = cursor.fetchone()
            
            if result:
                # Convert to dictionary
                columns = [col[0] for col in cursor.description]
                scan_dict = dict(zip(columns, result))
                
                # Parse indicators
                if scan_dict.get('indicators'):
                    import json
                    try:
                        scan_dict['indicators'] = json.loads(scan_dict['indicators'])
                    except:
                        scan_dict['indicators'] = {}
                
                logger.info(f"Retrieved latest scan for {'all pairs' if not pair else pair}")
                return scan_dict
            else:
                logger.info(f"No scans found for {'all pairs' if not pair else pair}")
                return None
        except Exception as e:
            logger.error(f"Error getting latest scan from database: {str(e)}")
            return None
    
    def get_latest_status(self):
        """
        Get the most recent bot status.
        
        Returns:
            dict: The latest bot status information, or None if not found
        """
        try:
            # Ensure connection is active
            if not self.conn:
                self.connect()
            
            cursor = self.conn.cursor()
            cursor.execute("SELECT * FROM bot_status ORDER BY timestamp DESC LIMIT 1")
            result = cursor.fetchone()
            
            if result:
                # Convert to dictionary
                columns = [col[0] for col in cursor.description]
                status_dict = dict(zip(columns, result))
                
                # Parse active_pairs
                if status_dict.get('active_pairs'):
                    status_dict['active_pairs_list'] = status_dict['active_pairs'].split(',')
                else:
                    status_dict['active_pairs_list'] = []
                
                logger.info("Retrieved latest bot status")
                return status_dict
            else:
                logger.info("No bot status entries found")
                return None
        except Exception as e:
            logger.error(f"Error getting latest status from database: {str(e)}")
            return None
    
    def import_from_csv(self, csv_file):
        """
        Import trades from a CSV file into the database.
        
        Args:
            csv_file (str): Path to the CSV file
            
        Returns:
            int: Number of trades imported
        """
        try:
            # Ensure connection is active
            if not self.conn:
                self.connect()
            
            # Read CSV file
            df = pd.read_csv(csv_file)
            
            # Check for required columns
            required_columns = ['timestamp', 'pair', 'action', 'price', 'quantity']
            for col in required_columns:
                if col not in df.columns:
                    logger.error(f"CSV file missing required column: {col}")
                    return 0
            
            # Make sure timestamp is formatted correctly
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Rename columns if needed to match database schema
            column_mapping = {
                'profit': 'net_profit',
                'profit_pct': 'profit_pct',
                'strategy': 'strategy',
                'order_id': 'order_id'
            }
            
            for old_col, new_col in column_mapping.items():
                if old_col in df.columns and old_col != new_col:
                    df[new_col] = df[old_col]
            
            # Add created_at column
            df['created_at'] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            
            # Convert DataFrame to records
            records = df.to_dict('records')
            
            # Insert into database
            cursor = self.conn.cursor()
            
            # Determine which columns to use from the dataframe
            columns = ['timestamp', 'pair', 'action', 'price', 'quantity', 'created_at']
            optional_columns = ['net_profit', 'profit_pct', 'order_id', 'strategy']
            
            for col in optional_columns:
                if col in df.columns:
                    columns.append(col)
            
            # Build the SQL query
            placeholders = ', '.join(['?'] * len(columns))
            columns_str = ', '.join(columns)
            
            query = f"INSERT INTO trades ({columns_str}) VALUES ({placeholders})"
            
            # Insert records
            count = 0
            for record in records:
                values = [str(record[col]) if col == 'timestamp' else record.get(col) for col in columns]
                cursor.execute(query, values)
                count += 1
            
            self.conn.commit()
            logger.info(f"Imported {count} trades from CSV file: {csv_file}")
            
            return count
        except Exception as e:
            logger.error(f"Error importing from CSV: {str(e)}")
            if self.conn:
                self.conn.rollback()
            return 0
    
    def export_to_csv(self, csv_file, start_date=None, end_date=None):
        """
        Export trades from the database to a CSV file.
        
        Args:
            csv_file (str): Path to the output CSV file
            start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to None.
            end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to None.
            
        Returns:
            int: Number of trades exported
        """
        try:
            # Get trades with filters
            df = self.get_trades(start_date=start_date, end_date=end_date)
            
            # Drop unwanted columns
            columns_to_keep = ['timestamp', 'pair', 'action', 'price', 'quantity', 
                               'net_profit', 'profit_pct', 'order_id', 'strategy']
            columns_to_keep = [col for col in columns_to_keep if col in df.columns]
            
            df = df[columns_to_keep]
            
            # Save to CSV
            df.to_csv(csv_file, index=False)
            logger.info(f"Exported {len(df)} trades to CSV file: {csv_file}")
            
            return len(df)
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            return 0
            
    def __del__(self):
        """Destructor to ensure the database connection is closed."""
        self.close()


# Singleton instance for global access
_db_instance = None

def get_db(db_file=None):
    """
    Get the global database instance.
    
    Args:
        db_file (str, optional): Database file path. Used only on first call.
        
    Returns:
        TradingDatabase: The database instance
    """
    global _db_instance
    
    if _db_instance is None:
        _db_instance = TradingDatabase(db_file)
    
    return _db_instance


# For testing
if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create database
    db = TradingDatabase("test_trading.db")
    
    # Test trade logging
    trade_id = db.log_trade(
        pair="BTCUSDT",
        action="BUY",
        price=50000.0,
        quantity=0.1,
        strategy="TEST"
    )
    
    # Test market scan logging
    scan_id = db.log_market_scan(
        pair="BTCUSDT",
        signal="BUY",
        price=50000.0,
        strategy="TEST",
        interval="1h",
        indicators={"rsi": 30, "macd": 0.5}
    )
    
    # Test bot status logging
    status_id = db.log_bot_status(
        status="RUNNING",
        account_value=10000.0,
        active_pairs=["BTCUSDT", "ETHUSDT"],
        message="Test status"
    )
    
    # Test queries
    trades = db.get_trades()
    print(f"Retrieved {len(trades)} trades")
    
    scans = db.get_market_scans()
    print(f"Retrieved {len(scans)} market scans")
    
    status = db.get_bot_status()
    print(f"Retrieved {len(status)} status entries")
    
    # Test latest queries
    latest_scan = db.get_latest_scan("BTCUSDT")
    print(f"Latest scan: {latest_scan}")
    
    latest_status = db.get_latest_status()
    print(f"Latest status: {latest_status}")
    
    # Close database
    db.close()