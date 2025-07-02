from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime, Text, Boolean
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import os

Base = declarative_base()

class PriceData(Base):
    __tablename__ = 'price_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(10), default='BTC')
    price = Column(Float, nullable=False)
    volume = Column(Float)
    high = Column(Float)
    low = Column(Float)
    open_price = Column(Float)
    close_price = Column(Float)
    market_cap = Column(Float)

class SentimentData(Base):
    __tablename__ = 'sentiment_data'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    source = Column(String(20))  # 'twitter' or 'reddit'
    post_id = Column(String(100), unique=True)
    text = Column(Text)
    sentiment_score = Column(Float)  # -1 to 1
    sentiment_label = Column(String(20))  # 'positive', 'negative', 'neutral'
    author = Column(String(100))
    likes = Column(Integer, default=0)
    retweets = Column(Integer, default=0)
    replies = Column(Integer, default=0)

class TechnicalAnalysis(Base):
    __tablename__ = 'technical_analysis'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    symbol = Column(String(10), default='BTC')
    timeframe = Column(String(10))  # '1h', '4h', '1d', etc.
    
    # Moving Averages
    sma_20 = Column(Float)
    sma_50 = Column(Float)
    sma_200 = Column(Float)
    ema_12 = Column(Float)
    ema_26 = Column(Float)
    
    # Oscillators
    rsi = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_histogram = Column(Float)
    stoch_k = Column(Float)
    stoch_d = Column(Float)
    williams_r = Column(Float)
    
    # Volatility
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    atr = Column(Float)
    
    # Volume
    obv = Column(Float)
    volume_sma = Column(Float)
    
    # Trend
    adx = Column(Float)
    cci = Column(Float)

class Predictions(Base):
    __tablename__ = 'predictions'
    
    id = Column(Integer, primary_key=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    prediction_for = Column(DateTime)  # When this prediction is for
    symbol = Column(String(10), default='BTC')
    
    # Predicted values
    predicted_price = Column(Float)
    confidence_score = Column(Float)  # 0 to 1
    prediction_type = Column(String(20))  # '1h', '4h', '24h', '7d'
    
    # Model info
    model_name = Column(String(50))
    model_version = Column(String(20))
    
    # Features used
    sentiment_weight = Column(Float)
    technical_weight = Column(Float)
    price_weight = Column(Float)
    
    # Actual outcome (filled later)
    actual_price = Column(Float)
    accuracy_score = Column(Float)
    is_correct = Column(Boolean)

class SystemLogs(Base):
    __tablename__ = 'system_logs'
    
    id = Column(Integer, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    level = Column(String(10))  # 'INFO', 'WARNING', 'ERROR'
    module = Column(String(50))
    message = Column(Text)
    details = Column(Text)

# Database setup
def get_database_url():
    return os.getenv('DATABASE_URL', 'sqlite:///bitcoin_analysis.db')

def create_database():
    engine = create_engine(get_database_url())
    Base.metadata.create_all(engine)
    return engine

def get_session():
    engine = create_database()
    Session = sessionmaker(bind=engine)
    return Session()

# Initialize database on import
if __name__ == "__main__":
    create_database()
    print("Database created successfully!")
