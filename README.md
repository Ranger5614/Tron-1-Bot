# Cryptocurrency Trading Bot

A beginner-friendly cryptocurrency trading bot using the Binance API, designed for users with low to medium risk tolerance interested in short-term/day trading.

## Features

- **Multiple Trading Strategies**: Simple Moving Average (SMA), Relative Strength Index (RSI), and Combined strategy
- **Comprehensive Risk Management**: Position sizing, stop-loss, take-profit, daily limits, and drawdown protection
- **Backtesting Framework**: Test strategies with historical data before trading
- **Paper Trading**: Simulate trading without risking real money
- **Monitoring and Alerting**: Track performance, receive alerts, and generate reports
- **Beginner-Friendly**: Designed for users with limited trading experience

## Project Structure

```
crypto_trading_bot/
├── config/
│   └── config.py           # Configuration parameters
├── data/                   # Data storage directory
│   └── reports/            # Generated reports
├── docs/                   # Documentation
│   ├── user_guide.md
│   ├── installation_guide.md
│   └── code_documentation.md
├── logs/                   # Log files
├── src/                    # Source code
│   ├── binance_api.py      # Binance API wrapper
│   ├── backtester.py       # Backtesting framework
│   ├── deploy.py           # Deployment and monitoring
│   ├── logger.py           # Logging setup
│   ├── paper_trading.py    # Paper trading functionality
│   ├── risk_manager.py     # Risk management
│   ├── strategies.py       # Trading strategies
│   ├── trading_bot.py      # Main bot implementation
│   ├── test_api.py         # API testing
│   ├── test_risk_management.py # Risk management testing
│   ├── test_strategies.py  # Strategy testing
│   └── run_tests.py        # Test runner
├── .env.template           # Template for .env file
└── README.md               # Project overview
```

## Quick Start

1. **Installation**:
   ```
   pip install python-binance pandas numpy matplotlib ccxt python-dotenv
   ```

2. **Configuration**:
   - Create a `.env` file with your Binance API keys
   - Adjust settings in `config/config.py`

3. **Paper Trading**:
   ```
   python src/paper_trading.py --duration 24 --interval 15
   ```

4. **Live Trading**:
   ```
   python src/deploy.py --check-interval 15
   ```

## Documentation

- [User Guide](docs/user_guide.md): Detailed instructions for using the bot
- [Installation Guide](docs/installation_guide.md): Setup instructions for different operating systems
- [Code Documentation](docs/code_documentation.md): Technical documentation of the codebase

## Safety Features

- Paper trading mode for risk-free testing
- Configurable risk parameters
- Daily loss limits
- Maximum drawdown protection
- Comprehensive error handling

## Disclaimer

This trading bot is provided for educational and informational purposes only. Cryptocurrency trading involves significant risk. Always start with paper trading and small amounts. Past performance is not indicative of future results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
