import os
from dotenv import load_dotenv

load_dotenv()

class Settings:
    # API Keys
    TWITTER_BEARER_TOKEN = os.getenv('TWITTER_BEARER_TOKEN')
    TWITTER_API_KEY = os.getenv('TWITTER_API_KEY')
    TWITTER_API_SECRET = os.getenv('TWITTER_API_SECRET')
    TWITTER_ACCESS_TOKEN = os.getenv('TWITTER_ACCESS_TOKEN')
    TWITTER_ACCESS_TOKEN_SECRET = os.getenv('TWITTER_ACCESS_TOKEN_SECRET')
    
    REDDIT_CLIENT_ID = os.getenv('REDDIT_CLIENT_ID')
    REDDIT_CLIENT_SECRET = os.getenv('REDDIT_CLIENT_SECRET')
    REDDIT_USER_AGENT = os.getenv('REDDIT_USER_AGENT', 'BitcoinAnalysisBot/1.0')
    
    BINANCE_API_KEY = os.getenv('BINANCE_API_KEY')
    BINANCE_SECRET_KEY = os.getenv('BINANCE_SECRET_KEY')
    
    COINGECKO_API_KEY = os.getenv('COINGECKO_API_KEY')
    
    TELEGRAM_BOT_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')
    
    # Database
    DATABASE_URL = os.getenv('DATABASE_URL', 'sqlite:///bitcoin_analysis.db')
    
    # Configuration
    UPDATE_INTERVAL_MINUTES = int(os.getenv('UPDATE_INTERVAL_MINUTES', 15))
    PREDICTION_DAYS = int(os.getenv('PREDICTION_DAYS', 7))
    MIN_SENTIMENT_POSTS = int(os.getenv('MIN_SENTIMENT_POSTS', 50))
    
    # Web
    PORT = int(os.getenv('PORT', 8000))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Crypto settings
    SYMBOL = 'BTC'
    FIAT_CURRENCY = 'USDT'
    TRADING_PAIR = f'{SYMBOL}{FIAT_CURRENCY}'
    
    # Social media search terms
    TWITTER_KEYWORDS = [
        'bitcoin', 'btc', '#bitcoin', '#btc', 
        'bitcoin price', 'btc price', 'bitcoin prediction',
        'bitcoin analysis', 'bitcoin chart', 'crypto',
        'cryptocurrency', 'hodl', 'satoshi'
    ]
    
    REDDIT_SUBREDDITS = [
        'Bitcoin', 'btc', 'CryptoCurrency', 'BitcoinMarkets',
        'CryptoMarkets', 'investing', 'SecurityAnalysis',
        'BitcoinBeginners', 'CryptoTechnology'
    ]
    
    # Technical analysis settings
    TA_TIMEFRAMES = ['1h', '4h', '1d', '1w']
    TA_INDICATORS = [
        'RSI', 'MACD', 'BB', 'SMA', 'EMA', 'STOCH',
        'WILLIAMS', 'ADX', 'CCI', 'ATR', 'OBV'
    ]
    
    # ML settings
    ML_FEATURES = [
        'price', 'volume', 'sentiment_score', 'sentiment_volume',
        'rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_50', 'sma_200',
        'ema_12', 'ema_26', 'volume_sma', 'price_change_pct'
    ]
    
    TRAIN_DAYS = 365  # Use 1 year of data for training
    PREDICTION_INTERVAL_HOURS = 24  # Predict next 24 hours

settings = Settings()
