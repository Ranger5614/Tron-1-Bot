# Cryptocurrency Trading Bot - Installation Guide

This guide provides detailed instructions for installing and setting up the cryptocurrency trading bot on different operating systems.

## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation on Linux](#installation-on-linux)
3. [Installation on macOS](#installation-on-macos)
4. [Installation on Windows](#installation-on-windows)
5. [Setting Up API Keys](#setting-up-api-keys)
6. [Verifying Installation](#verifying-installation)

## Prerequisites

Before installing the trading bot, ensure you have the following:

- Python 3.8 or higher
- pip (Python package installer)
- Git (optional, for cloning the repository)
- A Binance account
- Binance API key and secret with trading permissions

## Installation on Linux

### Step 1: Install Python and Required Packages

```bash
# Update package lists
sudo apt update

# Install Python and pip
sudo apt install python3 python3-pip python3-venv

# Install development tools
sudo apt install build-essential libssl-dev libffi-dev python3-dev
```

### Step 2: Clone or Download the Repository

```bash
# Using Git
git clone https://github.com/yourusername/crypto_trading_bot.git
cd crypto_trading_bot

# Or download and extract the ZIP file
# wget https://github.com/yourusername/crypto_trading_bot/archive/main.zip
# unzip main.zip
# cd crypto_trading_bot-main
```

### Step 3: Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

If requirements.txt is not available, install the packages manually:

```bash
pip install python-binance pandas numpy matplotlib ccxt python-dotenv
```

## Installation on macOS

### Step 1: Install Python and Required Packages

```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python
brew install python

# Ensure pip is up to date
pip3 install --upgrade pip
```

### Step 2: Clone or Download the Repository

```bash
# Using Git
git clone https://github.com/yourusername/crypto_trading_bot.git
cd crypto_trading_bot

# Or download and extract the ZIP file manually
```

### Step 3: Create and Activate Virtual Environment

```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

If requirements.txt is not available, install the packages manually:

```bash
pip install python-binance pandas numpy matplotlib ccxt python-dotenv
```

## Installation on Windows

### Step 1: Install Python

1. Download Python from [python.org](https://www.python.org/downloads/windows/)
2. Run the installer, ensuring you check "Add Python to PATH"
3. Verify installation by opening Command Prompt and typing:
   ```
   python --version
   pip --version
   ```

### Step 2: Clone or Download the Repository

```bash
# Using Git
git clone https://github.com/yourusername/crypto_trading_bot.git
cd crypto_trading_bot

# Or download and extract the ZIP file manually
```

### Step 3: Create and Activate Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

### Step 4: Install Dependencies

```bash
# Install required packages
pip install -r requirements.txt
```

If requirements.txt is not available, install the packages manually:

```bash
pip install python-binance pandas numpy matplotlib ccxt python-dotenv
```

## Setting Up API Keys

To use the trading bot, you need to set up API keys from Binance:

1. Log in to your Binance account
2. Navigate to API Management (usually found under your account or security settings)
3. Create a new API key (enable trading permissions, but disable withdrawal for security)
4. Copy the API key and secret key

After obtaining your API keys, create a `.env` file in the root directory of the project:

```
BINANCE_API_KEY=your_api_key_here
BINANCE_API_SECRET=your_api_secret_here
```

**Important Security Note**: Never share your API keys or commit them to version control. The `.env` file is included in `.gitignore` to prevent accidental commits.

## Verifying Installation

To verify that the installation was successful and the bot is working correctly:

1. Ensure your virtual environment is activated
2. Run the API test script:
   ```
   python src/test_api.py
   ```

3. If successful, you should see output confirming connection to Binance and retrieving market data.

4. Next, test the strategies:
   ```
   python src/test_strategies.py
   ```

5. This should generate backtest results and save them to the `data/` directory.

If all tests pass, your installation is complete and the bot is ready to use. Refer to the [User Guide](user_guide.md) for instructions on configuring and running the bot.
