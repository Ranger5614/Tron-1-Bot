import os
import logging
import config

# Ensure the log file directory exists
log_dir = os.path.dirname(config.LOG_FILE)
if log_dir and not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Create a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Create file handler to log to file
file_handler = logging.FileHandler(config.LOG_FILE)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

# Add file handler to logger
logger.addHandler(file_handler)

# Function to get the logger
def get_logger():
    return logger