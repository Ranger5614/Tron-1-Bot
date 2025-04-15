import os
import shutil
from datetime import datetime

def create_directory_structure():
    """Create the main directory structure for the project."""
    directories = [
        'src',
        'src/strategies',
        'src/utils',
        'src/api',
        'src/dashboard',
        'logs',
        'data',
        'data/backups'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"Created directory: {directory}")

def move_files():
    """Move files to their appropriate directories."""
    # Define file movements
    movements = {
        'src/strategies': [
            'strategies.py',
            'advanced_strategies.py',
            'bot_manager.py'
        ],
        'src/utils': [
            'logger.py',
            'logger_utils.py',
            'bot_monitor.py'
        ],
        'src/api': [
            'binance_api.py'
        ],
        'src/dashboard': [
            'dashboard.py'
        ]
    }
    
    # Move each file to its directory
    for directory, files in movements.items():
        for file in files:
            if os.path.exists(file):
                shutil.move(file, os.path.join(directory, file))
                print(f"Moved {file} to {directory}")

def organize_logs():
    """Organize log files into the logs directory."""
    # Create timestamp for backup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Move and rename log files
    log_files = [
        'bot.log',
        'api_errors.log',
        'debug.log',
        'errors.log',
        'latest_scan.txt'
    ]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            # Create backup in data/backups
            backup_path = os.path.join('data', 'backups', f'{log_file}_{timestamp}')
            shutil.copy2(log_file, backup_path)
            
            # Move to logs directory
            shutil.move(log_file, os.path.join('logs', log_file))
            print(f"Moved and backed up {log_file}")

def cleanup():
    """Remove unnecessary files."""
    files_to_remove = [
        'style.txt',  # Empty file
        'test.py',    # Test file
        'test2.py'    # Test file
    ]
    
    for file in files_to_remove:
        if os.path.exists(file):
            os.remove(file)
            print(f"Removed {file}")

def main():
    print("Starting project organization...")
    
    # Create directory structure
    create_directory_structure()
    
    # Move files to appropriate directories
    move_files()
    
    # Organize logs
    organize_logs()
    
    # Cleanup unnecessary files
    cleanup()
    
    print("\nProject organization complete!")
    print("\nNew structure:")
    print("├── src/")
    print("│   ├── strategies/")
    print("│   ├── utils/")
    print("│   ├── api/")
    print("│   └── dashboard/")
    print("├── logs/")
    print("├── data/")
    print("│   └── backups/")
    print("├── config.py")
    print("├── deploy.py")
    print("└── requirements.txt")

if __name__ == "__main__":
    main() 