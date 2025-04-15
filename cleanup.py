"""
Cleanup script to organize project structure and remove unnecessary files.
"""

import os
import shutil
from datetime import datetime
import config

def create_directory_structure():
    """Create organized directory structure."""
    directories = {
        'logs': config.LOG_DIR,
        'data': config.DATA_DIR,
        'src': {
            'strategies': os.path.join('src', 'strategies'),
            'utils': os.path.join('src', 'utils'),
            'api': os.path.join('src', 'api'),
            'dashboard': os.path.join('src', 'dashboard')
        }
    }
    
    # Create main directories
    for dir_path in [directories['logs'], directories['data']]:
        os.makedirs(dir_path, exist_ok=True)
        
    # Create src subdirectories
    for subdir in directories['src'].values():
        os.makedirs(subdir, exist_ok=True)
        
    return directories

def move_files_to_directories(directories):
    """Move files to their appropriate directories."""
    # Files to move to src/strategies
    strategy_files = [
        'strategies.py',
        'advanced_strategies.py',
        'risk_manager.py'
    ]
    
    # Files to move to src/utils
    util_files = [
        'logger.py',
        'logger_utils.py',
        'notifier.py',
        'database.py',
        'scan_logger.py',
        'scan_viewer.py'
    ]
    
    # Files to move to src/api
    api_files = [
        'binance_api.py'
    ]
    
    # Files to move to src/dashboard
    dashboard_files = [
        'dashboard.py',
        'style.css'
    ]
    
    # Move files
    for file in strategy_files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(directories['src']['strategies'], file))
            
    for file in util_files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(directories['src']['utils'], file))
            
    for file in api_files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(directories['src']['api'], file))
            
    for file in dashboard_files:
        if os.path.exists(file):
            shutil.move(file, os.path.join(directories['src']['dashboard'], file))

def cleanup_logs():
    """Clean up old log files."""
    # Move all log files to logs directory
    log_files = [
        'bot.log',
        'api_errors.log',
        'debug.log',
        'errors.log',
        'latest_scan.txt'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            # Create backup with timestamp
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_name = f"{os.path.splitext(log_file)[0]}_{timestamp}{os.path.splitext(log_file)[1]}"
            shutil.move(log_file, os.path.join(config.LOG_DIR, backup_name))

def remove_unnecessary_files():
    """Remove unnecessary files."""
    files_to_remove = [
        'style.txt',
        'RUN ME.txt'
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)

def main():
    """Main cleanup function."""
    print("Starting cleanup...")
    
    # Create directory structure
    print("Creating directory structure...")
    directories = create_directory_structure()
    
    # Move files to appropriate directories
    print("Moving files to appropriate directories...")
    move_files_to_directories(directories)
    
    # Clean up logs
    print("Cleaning up logs...")
    cleanup_logs()
    
    # Remove unnecessary files
    print("Removing unnecessary files...")
    remove_unnecessary_files()
    
    print("Cleanup complete!")

if __name__ == "__main__":
    main() 