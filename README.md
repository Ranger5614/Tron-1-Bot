# Crypto Trading Dashboard

A comprehensive dashboard for monitoring and analyzing cryptocurrency trading performance.

## Features

- Real-time trading metrics
- Performance analytics by trading pair
- Strategy performance comparison
- Trade history and analysis
- Interactive charts and visualizations

## Deployment to Streamlit Cloud

1. Fork this repository to your GitHub account
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Sign in with your GitHub account
4. Click "New app"
5. Select your forked repository
6. Set the main file path as: `src/dashboard/dashboard.py`
7. Click "Deploy"

## Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run src/dashboard/dashboard.py
```

## Data Sources

The dashboard supports multiple data sources:
- Database
- CSV files
- Sample data (for testing)

## Configuration

Create a `.env` file with your configuration:
```
DISCORD_WEBHOOK_URL=your_webhook_url
```

## License

MIT License
