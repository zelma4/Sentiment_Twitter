# 🚀 Bitcoin Sentiment Analysis Bot

Advanced Bitcoin sentiment analysis and prediction bot with ML features.

## ✨ Features

### Core Features
- **Real-time Data Collection**: Twitter, Reddit, CoinGecko price data
- **Advanced Sentiment Analysis**: RoBERTa/FinBERT models with VADER fallback
- **Technical Analysis**: 20+ indicators using TA-Lib
- **Price Prediction**: Multiple ML models (Random Forest, LightGBM, CNN-LSTM)
- **Multi-Horizon Predictions**: 1h, 4h, 24h forecasts with consensus
- **Telegram Alerts**: Smart alerts based on sentiment + technical signals

### Advanced ML Features
- **Neural Networks**: CNN-LSTM with attention mechanism
- **Feature Engineering**: Advanced features based on research
- **Feature Selection**: Boruta algorithm for optimal features
- **Model Explanability**: SHAP values for interpretability
- **Auto-Retraining**: Models retrain every 48h automatically
- **Backtesting**: PnL analysis, Sharpe ratio, drawdown metrics

### Data Sources
- **Social Media**: Twitter API v2, Reddit PRAW
- **Market Data**: CoinGecko, real-time price feeds
- **Enhanced Metrics**: Fear & Greed Index, Google Trends, StockTwits
- **On-chain Data**: Integration ready for blockchain metrics

## 🏗️ Architecture

```
main.py                 # Main bot orchestrator
├── analysis/
│   ├── advanced_sentiment.py      # RoBERTa/FinBERT sentiment
│   ├── advanced_predictor.py      # CNN-LSTM neural network
│   ├── advanced_feature_engineering.py # Research-based features
│   ├── lightgbm_predictor.py      # LightGBM model
│   ├── multi_horizon.py           # Multi-timeframe predictions
│   ├── backtesting.py             # Strategy backtesting
│   ├── technical_analyzer.py      # Technical indicators
│   └── predictor.py               # Base ML predictor
├── data_collectors/
│   ├── twitter_collector.py       # Twitter API integration
│   ├── reddit_collector.py        # Reddit API integration
│   ├── price_collector.py         # Price data collection
│   └── enhanced_collector.py      # Additional metrics
├── web/
│   ├── app.py                     # Flask dashboard
│   └── templates/                 # Web UI templates
├── database/
│   └── models.py                  # SQLAlchemy models
└── utils/
    └── helpers.py                 # Utility functions
```

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Clone repository
git clone <repo-url>
cd Sentiment_Twitter

# Create virtual environment
python -m venv venv_bitcoin
source venv_bitcoin/bin/activate  # Linux/Mac
# venv_bitcoin\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration
Create `.env` file:
```env
# Telegram
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Twitter API v2
TWITTER_BEARER_TOKEN=your_bearer_token

# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=your_user_agent

# Bot Settings
UPDATE_INTERVAL_MINUTES=10
CRYPTO_NAME=Bitcoin
SYMBOL=BTC
PREDICTION_DAYS=7
```

### 3. Run Bot
```bash
# Run locally
python main.py

# Run with dashboard
python main.py &  # Bot in background
# Dashboard available at http://localhost:8000
```

## 🐳 Docker Deployment

```bash
# Build and run
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f bitcoin-bot
```

## 📊 Dashboard

Web dashboard available at `http://localhost:8000` with:
- Real-time sentiment metrics
- Price predictions and charts
- Technical analysis signals
- Model performance metrics
- Advanced neural network predictions

## 🔧 Server Management

### Deploy to Server
```bash
# Deploy
./server_deploy.sh

# Check status
./server_check.sh

# Monitor
python monitor_bot.py

# Analyze signals
python signal_analyzer.py
```

### Train Models
```bash
# Initial model training
python train_server_models.py

# Quick model initialization
python quick_init_models.py
```

## 📈 Features Breakdown

### Sentiment Analysis
- **Models**: RoBERTa, FinBERT, VADER (fallback)
- **Sources**: Twitter, Reddit, StockTwits
- **Features**: Momentum, volatility, persistence, reversals
- **Research-based**: Optimized features from academic research

### ML Models
1. **Random Forest**: Baseline model
2. **LightGBM**: Gradient boosting with high performance
3. **CNN-LSTM**: Deep learning with attention mechanism
4. **Multi-Horizon**: Ensemble predictions across timeframes

### Technical Analysis
- **Indicators**: RSI, MACD, Bollinger Bands, SMA/EMA, Volume
- **Signals**: Automated buy/sell/hold recommendations
- **Strength**: Signal confidence scoring

### Alert System
- **Critical Alerts**: >5% price movement, strong sentiment
- **Regular Updates**: Moderate changes, trend shifts
- **Hourly Summaries**: Status updates every 2 hours
- **Smart Timing**: Rate limiting to prevent spam

## 🔬 Research Features

Based on academic research for cryptocurrency prediction:
- **Sentiment momentum** (highest importance feature)
- **Compound score** weighting
- **Negativity bias** correction  
- **Multi-horizon consensus** for robustness
- **Feature selection** using Boruta algorithm
- **Attention mechanisms** for temporal dependencies

## 📝 Log Analysis

```bash
# View bot logs
tail -f bitcoin_bot.log

# Check for errors
grep ERROR bitcoin_bot.log

# Monitor predictions
grep "PREDICTION" bitcoin_bot.log
```

## 🔄 Maintenance

### Model Retraining
- **Automatic**: Every 48 hours
- **Manual**: Run training scripts
- **Monitoring**: Check model performance logs

### Data Management
- **Database**: SQLite (local) or PostgreSQL (production)
- **Cleanup**: Old data automatically archived
- **Backup**: Daily backups recommended

## 📊 Performance Metrics

The bot tracks:
- **Prediction Accuracy**: MAPE, RMSE for price predictions
- **Sentiment Correlation**: Sentiment vs price movement correlation
- **Signal Quality**: Precision/recall for trading signals
- **Model Performance**: Individual model accuracy comparison

## 🚨 Troubleshooting

### Common Issues
1. **API Rate Limits**: Twitter/Reddit limits handled automatically
2. **Model Loading**: Neural models require sufficient RAM
3. **Dependencies**: Ensure all packages in requirements.txt installed
4. **Database**: Check SQLite permissions and disk space

### Support
Check logs for detailed error messages and consult documentation.

## 📜 License

This project is for educational and research purposes.

---

**⚠️ Disclaimer**: This bot is for informational purposes only. Do not use for actual trading without proper risk management.**
