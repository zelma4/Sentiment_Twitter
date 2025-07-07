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
    UPDATE_INTERVAL_MINUTES = int(os.getenv('UPDATE_INTERVAL_MINUTES', 10))  # Every 10 minutes
    PREDICTION_DAYS = int(os.getenv('PREDICTION_DAYS', 7))
    MIN_SENTIMENT_POSTS = int(os.getenv('MIN_SENTIMENT_POSTS', 50))
    
    # Web
    PORT = int(os.getenv('PORT', 8000))
    HOST = os.getenv('HOST', '0.0.0.0')
    
    # Crypto settings - CHANGE ONLY THIS LINE TO SWITCH COINS
    SYMBOL = 'BTC'  # Options: BTC, ETH, BNB, SOL, ADA, DOT, MATIC, AVAX, LINK, UNI, ATOM
    FIAT_CURRENCY = 'USDT'
    TRADING_PAIR = f'{SYMBOL}{FIAT_CURRENCY}'
    
    # Crypto configurations - автоматично підпасовуються під SYMBOL
    CRYPTO_CONFIGS = {
        'BTC': {
            'name': 'Bitcoin',
            'coingecko_id': 'bitcoin',
            'twitter_keywords': [
                'bitcoin', 'btc', '#bitcoin', '#btc', 
                'bitcoin price', 'btc price', 'bitcoin prediction',
                'bitcoin analysis', 'bitcoin chart', 'crypto',
                'cryptocurrency', 'hodl', 'satoshi', 'digital gold'
            ],
            'reddit_subreddits': [
                'Bitcoin', 'btc', 'CryptoCurrency', 'BitcoinMarkets',
                'CryptoMarkets', 'investing', 'SecurityAnalysis',
                'BitcoinBeginners', 'CryptoTechnology'
            ]
        },
        'ETH': {
            'name': 'Ethereum',
            'coingecko_id': 'ethereum',
            'twitter_keywords': [
                'ethereum', 'eth', '#ethereum', '#eth', 
                'ethereum price', 'eth price', 'ethereum prediction',
                'ethereum analysis', 'ethereum chart', 'crypto',
                'cryptocurrency', 'hodl', 'vitalik', 'defi',
                'smart contracts', 'gas fees', 'ethereum 2.0'
            ],
            'reddit_subreddits': [
                'ethereum', 'ethfinance', 'CryptoCurrency', 'ethtrader',
                'CryptoMarkets', 'investing', 'SecurityAnalysis',
                'defi', 'CryptoTechnology', 'ethstaker'
            ]
        },
        'BNB': {
            'name': 'Binance Coin',
            'coingecko_id': 'binancecoin',
            'twitter_keywords': [
                'binance', 'bnb', '#binance', '#bnb',
                'binance coin', 'bnb price', 'binance prediction',
                'binance analysis', 'bsc', 'binance smart chain',
                'cryptocurrency', 'hodl', 'cz'
            ],
            'reddit_subreddits': [
                'binance', 'BNBTrader', 'CryptoCurrency', 'BinanceSmartChain',
                'CryptoMarkets', 'investing', 'pancakeswap'
            ]
        },
        'SOL': {
            'name': 'Solana',
            'coingecko_id': 'solana',
            'twitter_keywords': [
                'solana', 'sol', '#solana', '#sol',
                'solana price', 'sol price', 'solana prediction',
                'solana analysis', 'phantom', 'solana nft',
                'cryptocurrency', 'hodl', 'defi solana'
            ],
            'reddit_subreddits': [
                'solana', 'SolanaTrader', 'CryptoCurrency', 'SolanaNFTs',
                'CryptoMarkets', 'investing', 'defi'
            ]
        },
        'ADA': {
            'name': 'Cardano',
            'coingecko_id': 'cardano',
            'twitter_keywords': [
                'cardano', 'ada', '#cardano', '#ada',
                'cardano price', 'ada price', 'cardano prediction',
                'cardano analysis', 'charles hoskinson', 'cardano smart contracts',
                'cryptocurrency', 'hodl', 'plutus'
            ],
            'reddit_subreddits': [
                'cardano', 'CardanoTraders', 'CryptoCurrency', 'CardanoStakePools',
                'CryptoMarkets', 'investing', 'CardanoDeveloper'
            ]
        },
        'DOT': {
            'name': 'Polkadot',
            'coingecko_id': 'polkadot',
            'twitter_keywords': [
                'polkadot', 'dot', '#polkadot', '#dot',
                'polkadot price', 'dot price', 'polkadot prediction',
                'polkadot analysis', 'substrate', 'parachain',
                'cryptocurrency', 'hodl', 'gavin wood'
            ],
            'reddit_subreddits': [
                'dot', 'polkadot_market', 'CryptoCurrency', 'Polkadot',
                'CryptoMarkets', 'investing', 'substrate'
            ]
        },
        'MATIC': {
            'name': 'Polygon',
            'coingecko_id': 'matic-network',
            'twitter_keywords': [
                'polygon', 'matic', '#polygon', '#matic',
                'polygon price', 'matic price', 'polygon prediction',
                'polygon analysis', 'layer 2', 'ethereum scaling',
                'cryptocurrency', 'hodl', 'defi polygon'
            ],
            'reddit_subreddits': [
                '0xPolygon', 'maticnetwork', 'CryptoCurrency', 'PolygonTraders',
                'CryptoMarkets', 'investing', 'defi'
            ]
        },
        'AVAX': {
            'name': 'Avalanche',
            'coingecko_id': 'avalanche-2',
            'twitter_keywords': [
                'avalanche', 'avax', '#avalanche', '#avax',
                'avalanche price', 'avax price', 'avalanche prediction',
                'avalanche analysis', 'subnet', 'avalanche defi',
                'cryptocurrency', 'hodl', 'emin gun sirer'
            ],
            'reddit_subreddits': [
                'Avax', 'AvaxTrader', 'CryptoCurrency', 'Avalanche',
                'CryptoMarkets', 'investing', 'defi'
            ]
        },
        'LINK': {
            'name': 'Chainlink',
            'coingecko_id': 'chainlink',
            'twitter_keywords': [
                'chainlink', 'link', '#chainlink', '#link',
                'chainlink price', 'link price', 'chainlink prediction',
                'chainlink analysis', 'oracle', 'smartcontract',
                'cryptocurrency', 'hodl', 'sergey nazarov'
            ],
            'reddit_subreddits': [
                'Chainlink', 'LINKTrader', 'CryptoCurrency', 'ChainlinkOfficial',
                'CryptoMarkets', 'investing', 'defi'
            ]
        },
        'UNI': {
            'name': 'Uniswap',
            'coingecko_id': 'uniswap',
            'twitter_keywords': [
                'uniswap', 'uni', '#uniswap', '#uni',
                'uniswap price', 'uni price', 'uniswap prediction',
                'uniswap analysis', 'defi', 'dex',
                'cryptocurrency', 'hodl', 'uniswap v3'
            ],
            'reddit_subreddits': [
                'UniSwap', 'uniswap', 'CryptoCurrency', 'UniswapTrader',
                'CryptoMarkets', 'investing', 'defi'
            ]
        },
        'ATOM': {
            'name': 'Cosmos',
            'coingecko_id': 'cosmos',
            'twitter_keywords': [
                'cosmos', 'atom', '#cosmos', '#atom',
                'cosmos price', 'atom price', 'cosmos prediction',
                'cosmos analysis', 'tendermint', 'cosmos hub',
                'cryptocurrency', 'hodl', 'ibc'
            ],
            'reddit_subreddits': [
                'cosmosnetwork', 'CosmosTraders', 'CryptoCurrency', 'CosmosAirdrops',
                'CryptoMarkets', 'investing', 'CosmosEcosystem'
            ]
        }
    }
    
    # Динамічне отримання конфігурації для поточного символу
    @property
    def current_crypto_config(self):
        return self.CRYPTO_CONFIGS.get(self.SYMBOL, self.CRYPTO_CONFIGS['BTC'])
    
    @property
    def CRYPTO_NAME(self):
        return self.current_crypto_config['name']
    
    @property
    def COINGECKO_ID(self):
        return self.current_crypto_config['coingecko_id']
    
    @property
    def TWITTER_KEYWORDS(self):
        return self.current_crypto_config['twitter_keywords']
    
    @property
    def REDDIT_SUBREDDITS(self):
        return self.current_crypto_config['reddit_subreddits']
    
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
