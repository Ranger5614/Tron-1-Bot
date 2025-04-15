"""
Enhanced logger utilities with database support for the trading bot.
"""

import csv
import os
import traceback
import logging
from datetime import datetime
import json
import sys

# Get a logger
logger = logging.getLogger("crypto_bot.logger_utils")

# Set up basic logging if not already configured
if not logger.handlers:
    # File handler for detailed logs
    try:
        logs_dir = "logs"
        if not os.path.exists(logs_dir):
            os.makedirs(logs_dir)
            
        file_handler = logging.FileHandler(os.path.join(logs_dir, "logger_utils.log"))
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        file_handler.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        
        # Console handler only for warnings and errors
        console_handler = logging.StreamHandler()
        console_formatter = logging.Formatter('%(levelname)s: %(message)s')
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(logging.WARNING)  # Only show WARNING and above on console
        logger.addHandler(console_handler)
        
        logger.setLevel(logging.INFO)
    except Exception as e:
        # Fallback basic handler if file logging setup fails
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        handler.setLevel(logging.WARNING)  # Only show WARNING and above
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.warning(f"Failed to set up file logging: {e}")

# Import database module with fallback to CSV if not available
try:
    from src.utils.database import get_db
    HAS_DATABASE = True
    logger.info("Database module imported successfully")
except ImportError:
    HAS_DATABASE = False
    logger.warning("Database module not found, falling back to CSV logging")

def log_trade_to_csv(pair, action, price, quantity, pnl=None, pnl_pct=None, conviction=None, filename=None):
    """
    Log a trade to CSV file with enhanced error handling and directory creation.
    If database module is available, also logs to the database.
    
    Args:
        pair (str): Trading pair (e.g., 'BTCUSDT')
        action (str): Trade action ('BUY' or 'SELL')
        price (float): Trade price
        quantity (float): Trade quantity
        pnl (float, optional): Profit/Loss in currency. Defaults to None.
        pnl_pct (float, optional): Profit/Loss percentage. Defaults to None.
        conviction (float, optional): Signal conviction level (0.0-1.0). Defaults to None.
        filename (str, optional): Output filename. Defaults to "trade_log.csv" in current directory.
    """
    # First, try to log to database if available
    if HAS_DATABASE:
        try:
            db = get_db()
            # Remove conviction parameter to avoid errors
            db_params = {
                'pair': pair,
                'action': action,
                'price': float(price),
                'quantity': float(quantity),
                'net_profit': float(pnl) if pnl is not None else None,
                'profit_pct': float(pnl_pct) if pnl_pct is not None else None
                # conviction is intentionally omitted
            }
            db.log_trade(**db_params)
            logger.info(f"Trade logged to database: {pair} {action} @ {price}")
        except Exception as e:
            logger.error(f"Error logging trade to database: {str(e)}")
    
    # Always log to CSV for backward compatibility
    try:
        # Set default filename if not provided
        if filename is None:
            # Use the correct bot directory path
            bot_dir = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "All Bots", "Crypto Bot")
            
            # Check if the directory exists
            if os.path.exists(bot_dir):
                filename = os.path.join(bot_dir, "trade_log.csv")
                logger.info(f"Using bot directory for trade log: {bot_dir}")
            else:
                # Fallback to current directory
                filename = "trade_log.csv"
                logger.warning(f"Bot directory not found: {bot_dir}, using current directory")
        
        # Print the absolute path to the file (only to log file, not console)
        abs_path = os.path.abspath(filename)
        logger.info(f"Absolute path of log file: {abs_path}")
        
        # Make sure directory exists
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        
        # Record attempt in log file only
        logger.info(f"Logging trade to {filename}: {pair} {action} @ {price}")
        
        # Check if file exists (log details only to file)
        file_exists = os.path.isfile(filename)
        logger.info(f"File exists: {file_exists}")
        if file_exists:
            file_size = os.path.getsize(filename)
            logger.info(f"File size: {file_size} bytes")
        
        # Ensure price and quantity are valid numbers
        try:
            price_value = float(price)
        except (ValueError, TypeError):
            price_value = 0.0
            logger.warning(f"Invalid price value: {price}, using 0.0")
        
        try:
            quantity_value = float(quantity)
        except (ValueError, TypeError):
            quantity_value = 0.0
            logger.warning(f"Invalid quantity value: {quantity}, using 0.0")
        
        # Format PnL values
        pnl_value = round(float(pnl), 6) if pnl is not None else ""
        pnl_pct_value = round(float(pnl_pct), 4) if pnl_pct is not None else ""
        
        # Format conviction value
        conviction_value = round(float(conviction), 3) if conviction is not None else ""
        
        # Get current UTC timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Construct the row data (only log to file, not console)
        row_data = [
            timestamp,
            pair,
            action,
            round(price_value, 6),
            round(quantity_value, 6),
            pnl_value,
            pnl_pct_value,
            conviction_value
        ]
        logger.info(f"About to write row: {row_data}")
        
        # Open the file and write the data
        with open(filename, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header if file is new
            if not file_exists:
                writer.writerow(["timestamp", "pair", "action", "price", "quantity", "net_profit", "profit_pct", "conviction"])
            
            # Write the trade data
            writer.writerow(row_data)
        
        # Message important enough to show on console (summary of what happened)
        if action == 'BUY':
            logger.warning(f"BUY TRADE: {pair} @ ${price:.2f}, quantity: {quantity}")
        elif action == 'SELL':
            if pnl is not None and pnl_pct is not None:
                profit_msg = f"P/L: ${pnl:.2f} ({pnl_pct:.2f}%)" 
                logger.warning(f"SELL TRADE: {pair} @ ${price:.2f}, quantity: {quantity}, {profit_msg}")
            else:
                logger.warning(f"SELL TRADE: {pair} @ ${price:.2f}, quantity: {quantity}")
        else:
            # Still log to file but details only go to log file
            logger.info(f"Successfully logged trade to CSV: {filename}")
        
        # Also create another copy in root directory for dashboard 
        # to ensure it can be found regardless of configuration
        with open("trade_log.csv", mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Write header if file is new
            if not os.path.isfile("trade_log.csv") or os.path.getsize("trade_log.csv") == 0:
                writer.writerow(["timestamp", "pair", "action", "price", "quantity", "net_profit", "profit_pct", "conviction"])
            
            # Write the trade data
            writer.writerow(row_data)
        
        # Log file details for debugging (only to log file)
        csv_exists = os.path.exists("trade_log.csv")
        csv_size = os.path.getsize("trade_log.csv") if csv_exists else 0
        logger.info(f"Root CSV check - exists: {csv_exists}, size: {csv_size} bytes")
        
        return True
    except Exception as e:
        error_msg = f"Error logging trade to CSV: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        # Try to log to a fallback file
        try:
            with open("trade_log_error.txt", "a") as f:
                f.write(f"\n\n===== {datetime.utcnow()} =====\n")
                f.write(f"Error logging trade: {str(e)}\n")
                f.write(f"Attempted to log: {pair} {action} @ {price}, qty={quantity}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
        except Exception as inner_e:
            logger.error(f"Also failed to write error log: {inner_e}")
        
        return False

def test_trade_logging():
    """
    Test function to diagnose trade logging issues.
    This will attempt to create test entries in the trade log.
    """
    print("\n===== TESTING TRADE LOGGING =====")
    
    # Print current working directory
    cwd = os.getcwd()
    print(f"Current working directory: {cwd}")
    
    # Check if we can write to the current directory
    try:
        test_file = os.path.join(cwd, "test_write_permission.txt")
        with open(test_file, "w") as f:
            f.write("Test write permission")
        os.remove(test_file)
        print("✅ Directory is writable")
    except Exception as e:
        print(f"❌ Directory is NOT writable: {e}")
    
    # Check if trade_log.csv exists and its properties
    log_file = os.path.join(cwd, "trade_log.csv")
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f"✅ trade_log.csv exists, size: {size} bytes")
        
        # Check if file is not locked
        try:
            with open(log_file, "a") as f:
                pass  # Just testing if we can open it for append
            print("✅ trade_log.csv is not locked")
        except Exception as e:
            print(f"❌ trade_log.csv is locked: {e}")
    else:
        print("❓ trade_log.csv does not exist yet")
    
    # Try to create a test trade entry
    try:
        result = log_trade_to_csv(
            pair="TEST/USDT",
            action="TEST",
            price=1000.0,
            quantity=0.01,
            pnl=0.5,
            pnl_pct=5.0,
            conviction=0.75
        )
        if result:
            print("✅ Test trade successfully logged")
        else:
            print("❌ Test trade logging returned False")
    except Exception as e:
        print(f"❌ Error during test trade logging: {e}")
        print(traceback.format_exc())
    
    # Check if the file exists after our test
    if os.path.exists(log_file):
        size = os.path.getsize(log_file)
        print(f"✅ After test: trade_log.csv exists, size: {size} bytes")
        
        # Read the first few lines to verify content
        try:
            with open(log_file, "r") as f:
                lines = f.readlines()[:5]  # Read up to 5 lines
            print("File content preview:")
            for line in lines:
                print(f"  {line.strip()}")
        except Exception as e:
            print(f"❌ Could not read file: {e}")
    else:
        print("❌ After test: trade_log.csv still does not exist")
    
    print("===== TEST COMPLETE =====\n")
    return True

def log_latest_scan(symbol, signal, price, indicators=None, strategy=None, interval=None):
    """
    Write the latest scan for a symbol to the database (if available) and a simple text file.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTCUSDT')
        signal (str): Signal detected (e.g., 'BUY', 'SELL', 'HOLD')
        price (float): Current price
        indicators (dict): Dictionary of indicator values
        strategy (str): Strategy used
        interval (str): Timeframe interval
    """
    # First, try to log to database if available
    if HAS_DATABASE:
        try:
            db = get_db()
            db.log_market_scan(
                pair=symbol,
                signal=signal,
                price=float(price) if price is not None else 0.0,
                strategy=strategy,
                interval=interval,
                indicators=indicators
            )
            logger.info(f"Market scan logged to database: {symbol} {signal} @ {price}")
        except Exception as e:
            logger.error(f"Error logging market scan to database: {str(e)}")
    
    # File to store the latest scan
    scan_file = "latest_scan.txt"
    
    try:
        # Log to the application log (file only)
        logger.info(f"Logging scan for {symbol} to {scan_file}")
        
        # Format timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Format signal with emoji
        signal_icon = "⚪"  # default/hold
        if isinstance(signal, str) and signal.upper() == 'BUY':
            signal_icon = "🟢"
        elif isinstance(signal, str) and signal.upper() == 'SELL':
            signal_icon = "🔴"
        elif isinstance(signal, str) and signal.upper() == 'ERROR':
            signal_icon = "⚠️"
        
        # Create content
        lines = [
            "====== LATEST MARKET SCAN ======",
            f"Time: {timestamp}",
            f"Pair: {symbol}",
            f"Signal: {signal_icon} {signal}",
        ]
        
        # Add price (with error handling)
        try:
            lines.append(f"Price: ${float(price):.4f}")
        except (ValueError, TypeError):
            lines.append(f"Price: {price}")
        
        if interval:
            lines.append(f"Interval: {interval}")
        
        if strategy:
            lines.append(f"Strategy: {strategy}")
        
        # Check for conviction level in indicators
        conviction = None
        if indicators and isinstance(indicators, dict) and 'signal_strength' in indicators:
            conviction = indicators['signal_strength']
            lines.append(f"Conviction: {conviction:.2f}")
        
        # Add indicators if provided
        if indicators and isinstance(indicators, dict):
            lines.append("\nIndicators:")
            for name, value in indicators.items():
                try:
                    if isinstance(value, float):
                        lines.append(f"  {name}: {value:.4f}")
                    else:
                        lines.append(f"  {name}: {value}")
                except Exception as e:
                    lines.append(f"  {name}: Error formatting value - {str(e)}")
        
        # Add API diagnostic information if this is an ERROR signal
        if isinstance(signal, str) and signal.upper() == 'ERROR':
            lines.append("\nDiagnostic Information:")
            lines.append(f"Python version: {sys.version}")
            
            # Add memory usage if psutil is available
            try:
                import psutil
                process = psutil.Process(os.getpid())
                memory_usage = process.memory_info().rss / 1024 / 1024  # in MB
                lines.append(f"Memory usage: {memory_usage:.2f} MB")
            except ImportError:
                lines.append("Memory usage: psutil not available")
            
            # Add network diagnostic info
            try:
                import socket
                hostname = socket.gethostname()
                local_ip = socket.gethostbyname(hostname)
                lines.append(f"Hostname: {hostname}")
                lines.append(f"Local IP: {local_ip}")
            except:
                lines.append("Network info: Error retrieving")
            
            # If we have error details in indicators, show them prominently
            if indicators and isinstance(indicators, dict) and 'error' in indicators:
                lines.append(f"\nERROR DETAILS: {indicators['error']}")
        
        lines.append("\n==============================")
        
        # Debug log the content (file only)
        logger.info(f"Writing {len(lines)} lines to {scan_file}")
        
        # Write to file (overwriting previous content)
        with open(scan_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        logger.info(f"Successfully wrote to {scan_file}")
        
        # Create a separate error log file for errors to track history
        if isinstance(signal, str) and signal.upper() == 'ERROR':
            error_log_file = f"error_log_{symbol}.txt"
            with open(error_log_file, 'a', encoding='utf-8') as f:
                f.write(f"\n\n===== {timestamp} =====\n")
                f.write('\n'.join(lines))
                f.write("\n\n")
            logger.info(f"Error details also written to {error_log_file}")
            
        # For BUY/SELL signals, show on console
        if signal == 'BUY':
            logger.warning(f"BUY SIGNAL: {symbol} @ ${price:.2f}")
        elif signal == 'SELL':
            logger.warning(f"SELL SIGNAL: {symbol} @ ${price:.2f}")
            
    except Exception as e:
        # Log the detailed error with traceback
        error_msg = f"Error in log_latest_scan: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_msg)
        
        # Try to write the error to the file so we can see what's happening
        try:
            with open(scan_file, 'w', encoding='utf-8') as f:
                f.write(f"ERROR LOGGING SCAN: {str(e)}\n\n")
                f.write(f"Symbol: {symbol}\n")
                f.write(f"Signal: {signal}\n")
                f.write(f"Price: {price}\n")
                f.write(f"Strategy: {strategy}\n")
                f.write(f"Traceback:\n{traceback.format_exc()}")
        except:
            pass

def log_debug_information(symbol, message, data=None, filename=None):
    """
    Log detailed debugging information for a specific symbol.
    Useful for tracking down issues with specific trading pairs.
    
    Args:
        symbol (str): Trading pair (e.g., 'BTCUSDT')
        message (str): Debug message
        data (dict, optional): Additional data to log. Defaults to None.
        filename (str, optional): Custom debug log filename. Defaults to "debug_{symbol}.log".
    """
    if filename is None:
        filename = f"debug_{symbol}.log"
    
    try:
        # Format timestamp
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log to regular log as well (debug level - won't appear in console)
        logger.debug(f"DEBUG {symbol}: {message}")
        
        # Create content
        lines = [f"===== DEBUG LOG: {timestamp} =====", f"Symbol: {symbol}", f"Message: {message}"]
        
        # Add stack trace for context
        stack = traceback.format_stack()[:-1]  # Exclude this function call
        lines.append("\nStack Trace:")
        lines.extend([f"  {line.strip()}" for line in stack])
        
        # Add additional data if provided
        if data:
            lines.append("\nAdditional Data:")
            if isinstance(data, dict):
                for key, value in data.items():
                    try:
                        if isinstance(value, dict):
                            lines.append(f"  {key}:")
                            for k, v in value.items():
                                lines.append(f"    {k}: {v}")
                        else:
                            lines.append(f"  {key}: {value}")
                    except:
                        lines.append(f"  {key}: Error formatting value")
            else:
                lines.append(f"  {data}")
        
        lines.append("\n")
        
        # Write to file (appending)
        with open(filename, 'a', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        # Keep file size manageable (max 1MB)
        if os.path.getsize(filename) > 1024 * 1024:
            with open(filename, 'r', encoding='utf-8') as f:
                content = f.readlines()
            
            # Keep only the last half of the file
            with open(filename, 'w', encoding='utf-8') as f:
                f.writelines(content[len(content)//2:])
        
        return True
    except Exception as e:
        logger.error(f"Error logging debug information: {str(e)}")
        return False

def log_bot_status(status, account_value=None, active_pairs=None, message=None):
    """
    Log the bot's status to the database (if available).
    
    Args:
        status (str): Bot status (e.g., 'RUNNING', 'STOPPED', 'ERROR')
        account_value (float, optional): Current account value. Defaults to None.
        active_pairs (list, optional): List of active trading pairs. Defaults to None.
        message (str, optional): Optional status message. Defaults to None.
    """
    if HAS_DATABASE:
        try:
            db = get_db()
            db.log_bot_status(
                status=status,
                account_value=account_value,
                active_pairs=active_pairs,
                message=message
            )
            logger.info(f"Bot status logged to database: {status}")
            return True
        except Exception as e:
            logger.error(f"Error logging bot status to database: {str(e)}")
            return False
    else:
        logger.warning("Database not available for logging bot status")
        return False

def log_api_error(symbol, endpoint, error_message, additional_data=None):
    """
    Log API errors for better debugging of connection issues.
    
    Args:
        symbol (str): Trading pair involved (e.g., 'BTCUSDT')
        endpoint (str): API endpoint that failed
        error_message (str): Error message
        additional_data (dict, optional): Any additional relevant data
    """
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    
    # Log to regular logger - this should show on console since it's an error
    logger.error(f"API ERROR - {symbol} - {endpoint}: {error_message}")
    
    # Write to dedicated API error log
    api_error_file = "api_errors.log"
    
    try:
        with open(api_error_file, 'a', encoding='utf-8') as f:
            f.write(f"\n===== {timestamp} =====\n")
            f.write(f"Symbol: {symbol}\n")
            f.write(f"Endpoint: {endpoint}\n")
            f.write(f"Error: {error_message}\n")
            
            if additional_data:
                f.write("Additional data:\n")
                if isinstance(additional_data, dict):
                    for key, value in additional_data.items():
                        f.write(f"  {key}: {value}\n")
                else:
                    f.write(f"  {additional_data}\n")
            
            f.write("\n")
        
        return True
    except Exception as e:
        logger.error(f"Error logging API error: {str(e)}")
        return False

def import_trades_from_csv(csv_file):
    """
    Import trades from a CSV file into the database.
    
    Args:
        csv_file (str): Path to the CSV file
        
    Returns:
        int: Number of trades imported, or -1 if database not available
    """
    if not HAS_DATABASE:
        logger.warning("Database not available for importing trades")
        return -1
    
    try:
        db = get_db()
        count = db.import_from_csv(csv_file)
        logger.info(f"Imported {count} trades from {csv_file} to database")
        return count
    except Exception as e:
        logger.error(f"Error importing trades from CSV: {str(e)}")
        return 0

def export_trades_to_csv(csv_file, start_date=None, end_date=None):
    """
    Export trades from the database to a CSV file.
    
    Args:
        csv_file (str): Path to the output CSV file
        start_date (str, optional): Start date in format 'YYYY-MM-DD'. Defaults to None.
        end_date (str, optional): End date in format 'YYYY-MM-DD'. Defaults to None.
        
    Returns:
        int: Number of trades exported, or -1 if database not available
    """
    if not HAS_DATABASE:
        logger.warning("Database not available for exporting trades")
        return -1
    
    try:
        db = get_db()
        count = db.export_to_csv(csv_file, start_date, end_date)
        logger.info(f"Exported {count} trades to {csv_file} from database")
        return count
    except Exception as e:
        logger.error(f"Error exporting trades to CSV: {str(e)}")
        return 0

# For testing
if __name__ == "__main__":
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Run the test function
    test_trade_logging()
    
    # Test the logging functions
    log_latest_scan(
        symbol="BTCUSDT",
        signal="BUY",
        price=42350.75,
        indicators={
            "rsi": 28.5,
            "short_ma": 42100.50,
            "long_ma": 41950.25,
            "signal_strength": 0.85  # TRON 1.1 conviction level
        },
        strategy="TRON11",
        interval="10m"
    )
    
    # Test trade logging with conviction level
    log_trade_to_csv(
        pair="BTCUSDT",
        action="BUY",
        price=42350.75,
        quantity=0.001,
        conviction=0.85
    )
    
    # Test API error logging
    log_api_error(
        symbol="BTCUSDT",
        endpoint="get_ticker_price",
        error_message="Connection refused",
        additional_data={"attempts": 3, "timeout": 5000}
    )
    
    # Test debug logging
    log_debug_information(
        symbol="BTCUSDT",
        message="Testing debug log",
        data={"price": 42350.75, "volume": 1250.5}
    )
    
    print("Test complete - check latest_scan.txt and trade_log.csv")