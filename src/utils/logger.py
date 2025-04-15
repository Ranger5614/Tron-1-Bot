"""
Centralized logging configuration for the trading bot.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import config

# Create logs directory if it doesn't exist
os.makedirs(config.LOG_DIR, exist_ok=True)

# Log file paths
LOG_FILES = {
    'main': os.path.join(config.LOG_DIR, 'bot.log'),
    'trades': os.path.join(config.LOG_DIR, 'trades.log'),
    'errors': os.path.join(config.LOG_DIR, 'errors.log'),
    'api': os.path.join(config.LOG_DIR, 'api.log'),
    'scans': os.path.join(config.LOG_DIR, 'scans.log')
}

# Maximum log file size (10MB)
MAX_LOG_SIZE = 10 * 1024 * 1024
# Number of backup files to keep
BACKUP_COUNT = 5

def setup_logger(name, log_file, level=logging.INFO):
    """Setup a logger with rotation."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create handlers
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=MAX_LOG_SIZE,
        backupCount=BACKUP_COUNT
    )
    console_handler = logging.StreamHandler()
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )
    
    # Add formatters to handlers
    file_handler.setFormatter(file_formatter)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

# Initialize loggers
main_logger = setup_logger('bot', LOG_FILES['main'])
trade_logger = setup_logger('trades', LOG_FILES['trades'])
error_logger = setup_logger('errors', LOG_FILES['errors'], logging.ERROR)
api_logger = setup_logger('api', LOG_FILES['api'])
scan_logger = setup_logger('scans', LOG_FILES['scans'])

def get_logger(logger_type='main'):
    """Get the appropriate logger based on type."""
    loggers = {
        'main': main_logger,
        'trades': trade_logger,
        'errors': error_logger,
        'api': api_logger,
        'scans': scan_logger
    }
    return loggers.get(logger_type, main_logger)

def console_message(message):
    """Print formatted console message."""
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")

def cleanup_old_logs():
    """Clean up old log files."""
    for log_file in LOG_FILES.values():
        if os.path.exists(log_file):
            # Keep only the last BACKUP_COUNT files
            base_name = os.path.basename(log_file)
            dir_name = os.path.dirname(log_file)
            
            # Remove old backup files
            for i in range(BACKUP_COUNT + 1, 10):  # Check up to .9
                old_file = f"{log_file}.{i}"
                if os.path.exists(old_file):
                    try:
                        os.remove(old_file)
                    except Exception as e:
                        console_message(f"Error removing old log file {old_file}: {e}")

# Clean up old logs on import
cleanup_old_logs()

# Add a startup banner showing the bot is running
def show_startup_banner():
    """Show a clean startup banner when the bot runs"""
    print("\n" + "="*50)
    print("               CRYPTO TRADING BOT")
    print("                   TRON 1.1")
    print("="*50)
    print("\n💰 Bot is running. Minimal console output enabled.")
    print("   Check logs for detailed information.\n")
    
    # If we're in test mode, show that prominently
    if hasattr(config, 'USE_TESTNET') and config.USE_TESTNET:
        print("⚠️  TESTNET MODE - No real trades will be executed")
    
    print("\nPress Ctrl+C to stop the bot\n")