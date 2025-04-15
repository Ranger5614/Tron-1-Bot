#!/usr/bin/env python
"""
Run the HYPERION Trading System Dashboard
"""

import os
import sys
import subprocess
import webbrowser
import time
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("dashboard_runner")

def run_dashboard():
    """Run the Streamlit dashboard"""
    try:
        # Get the path to the dashboard script
        dashboard_path = os.path.join("src", "dashboard", "dashboard.py")
        
        # Check if the dashboard file exists
        if not os.path.exists(dashboard_path):
            logger.error(f"Dashboard file not found: {dashboard_path}")
            return False
        
        # Open the dashboard in the default browser after a short delay
        def open_browser():
            time.sleep(2)  # Wait for Streamlit to start
            webbrowser.open("http://localhost:8501")
        
        # Start the browser in a separate thread
        import threading
        threading.Thread(target=open_browser).start()
        
        # Run the Streamlit app with the --server.headless flag to bypass the welcome message
        logger.info("Starting HYPERION Trading System Dashboard...")
        subprocess.run(["python", "-m", "streamlit", "run", dashboard_path, "--server.headless", "true"])
        
        return True
    except Exception as e:
        logger.error(f"Error running dashboard: {e}")
        return False

if __name__ == "__main__":
    # Add the current directory to the path
    sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))
    
    # Run the dashboard
    run_dashboard() 