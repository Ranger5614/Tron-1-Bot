# Cryptocurrency Trading Bot - User Guide

## Introduction

This cryptocurrency trading bot is designed for beginners with low to medium risk tolerance who want to engage in short-term/day trading on Binance. The bot implements multiple trading strategies, comprehensive risk management, and monitoring capabilities to help you trade cryptocurrencies more effectively.

## Table of Contents

1. [Installation](#installation)
2. [Configuration](#configuration)
3. [Trading Strategies](#trading-strategies)
4. [Risk Management](#risk-management)
5. [Paper Trading](#paper-trading)
6. [Live Trading](#live-trading)
7. [Monitoring and Alerts](#monitoring-and-alerts)
8. [Troubleshooting](#troubleshooting)
9. [FAQ](#faq)

## Installation

### Prerequisites

- Python 3.8 or higher
- Binance account
- Binance API key and secret (with trading permissions)

### Setup

1. Clone or download the repository to your local machine.

2. Install the required dependencies:
   ```
   pip install python-binance pandas numpy matplotlib ccxt python-dotenv
   ```

3. Create a `.env` file in the root directory based on the provided `.env.template`:
   ```
   BINANCE_API_KEY=your_api_key_here
   BINANCE_API_SECRET=your_api_secret_here
   ```

## Configuration

The main configuration file is located at `config/config.py`. Here you can customize various aspects of the bot:

### API Configuration
```python
API_KEY = os.getenv('BINANCE_API_KEY', '')
API_SECRET = os.getenv('BINANCE_API_SECRET', '')
```

### Trading Parameters
```python
TRADING_PAIRS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT']  # Trading pairs
TRADE_QUANTITY = 0.001  # Default trade quantity in BTC
USE_TESTNET = True  # Use Binance testnet for paper trading
```

### Strategy Parameters
```python
STRATEGY = 'SMA'  # Strategy options: 'SMA', 'RSI', 'COMBINED'
SHORT_WINDOW = 9  # Short moving average window
LONG_WINDOW = 21  # Long moving average window
RSI_PERIOD = 14  # RSI indicator period
RSI_OVERBOUGHT = 70  # RSI overbought threshold
RSI_OVERSOLD = 30  # RSI oversold threshold
```

### Risk Management
```python
MAX_TRADES_PER_DAY = 5  # Maximum number of trades per day
STOP_LOSS_PERCENTAGE = 2.0  # Stop loss percentage
TAKE_PROFIT_PERCENTAGE = 4.0  # Take profit percentage
MAX_RISK_PER_TRADE = 1.0  # Maximum risk per trade (percentage of account)
DAILY_LOSS_LIMIT = 5.0  # Daily loss limit (percentage of account)
```

## Trading Strategies

The bot implements three trading strategies:

### Simple Moving Average (SMA)
This strategy generates buy signals when the short-term moving average crosses above the long-term moving average, and sell signals when the short-term moving average crosses below the long-term moving average.

### Relative Strength Index (RSI)
This strategy generates buy signals when the RSI falls below the oversold threshold and sell signals when the RSI rises above the overbought threshold.

### Combined Strategy
This strategy combines both SMA and RSI indicators. It generates buy signals when both strategies indicate a buy, and sell signals when both indicate a sell. If the strategies disagree, it holds the current position.

## Risk Management

The bot includes several risk management features to protect your investment:

### Position Sizing
The bot calculates position size based on your risk tolerance, ensuring you never risk more than a specified percentage of your account on a single trade.

### Stop-Loss and Take-Profit
Each trade automatically includes stop-loss and take-profit orders to limit potential losses and secure profits.

### Daily Limits
The bot enforces daily limits on the number of trades and maximum loss to prevent excessive trading and significant losses in a single day.

### Maximum Drawdown Protection
The bot monitors your account's drawdown and can stop trading if it exceeds a specified threshold.

## Paper Trading

Before trading with real money, it's recommended to use paper trading to test the bot's performance:

1. Ensure `USE_TESTNET = True` in the configuration file.

2. Run the paper trading script:
   ```
   python src/paper_trading.py --duration 24 --interval 15
   ```
   This will run paper trading for 24 hours, checking for signals every 15 minutes.

3. Analyze the results in the `data/` directory, which will contain CSV files with trade history and performance metrics.

## Live Trading

Once you're satisfied with the paper trading results, you can switch to live trading:

1. Set `USE_TESTNET = False` in the configuration file.

2. Deploy the bot:
   ```
   python src/deploy.py --check-interval 15
   ```
   This will start the bot with monitoring checks every 15 minutes.

3. The bot will run continuously, executing trades based on your configured strategy and risk parameters.

## Monitoring and Alerts

The bot includes a comprehensive monitoring system:

### Performance Tracking
The bot tracks your account value, active trades, and overall performance over time.

### Alerts
The bot can generate alerts for various conditions:
- Excessive drawdown
- Daily loss limit exceeded
- High error rate

### Reports
The bot generates HTML reports with detailed performance metrics, trade history, and alerts. These reports are saved in the `data/reports/` directory.

### Email Notifications
To enable email notifications for alerts, configure the email settings in the `deploy.py` file:
```python
email_config = {
    'enabled': True,
    'smtp_server': 'smtp.gmail.com',
    'smtp_port': 587,
    'username': 'your_email@gmail.com',
    'password': 'your_app_password',
    'from_email': 'your_email@gmail.com',
    'to_email': 'recipient_email@example.com'
}
```

## Troubleshooting

### API Connection Issues
- Verify your API keys are correct in the `.env` file
- Ensure your API keys have the necessary permissions for trading
- Check your internet connection

### Trading Issues
- Ensure you have sufficient funds in your Binance account
- Check that the trading pairs you've configured are valid
- Verify the minimum trade quantity for your selected pairs

### Bot Not Executing Trades
- Check the logs for error messages
- Verify that your strategy is generating signals
- Ensure risk management rules are not preventing trades

## FAQ

### How much money do I need to start?
You can start with as little as $100, but it's recommended to have at least $500 for more effective position sizing.

### Which cryptocurrency pairs should I trade?
Start with major pairs like BTCUSDT, ETHUSDT, and BNBUSDT, which have higher liquidity and lower spreads.

### How do I know which strategy is best?
Use the backtesting functionality to compare different strategies with historical data:
```
python src/test_strategies.py
```

### Is this bot profitable?
The bot's profitability depends on market conditions, your chosen strategy, and risk parameters. Always start with paper trading to evaluate performance before using real money.

### How can I customize the strategies?
You can modify the strategy parameters in the configuration file or create new strategies by extending the base Strategy class in `src/strategies.py`.

### How do I update the bot?
Regularly check for updates in the repository and pull the latest changes. After updating, test with paper trading before resuming live trading.

### What if I want to stop the bot?
You can stop the bot at any time by pressing Ctrl+C in the terminal where it's running. The bot will generate a final report before shutting down.
