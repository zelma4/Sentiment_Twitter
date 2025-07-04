#!/usr/bin/env python3
"""
Quick fix script to test the dashboard locally and debug issues.
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from web.app import create_app
from database.models import get_session, PriceData, SentimentData


def test_dashboard_apis():
    """Test the dashboard APIs locally"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🔍 Testing dashboard APIs...")
    
    # Create Flask app
    app = create_app()
    
    # Test sentiment summary endpoint
    with app.test_client() as client:
        logger.info("Testing /api/sentiment-summary...")
        response = client.get('/api/sentiment-summary')
        print(f"Status: {response.status_code}")
        print(f"Response: {response.get_json()}")
        
        logger.info("Testing /api/price-data...")
        response = client.get('/api/price-data')
        print(f"Status: {response.status_code}")
        print(f"Response: {response.get_json()}")
        
        logger.info("Testing /api/stats...")
        response = client.get('/api/stats')
        print(f"Status: {response.status_code}")
        print(f"Response: {response.get_json()}")


def check_database_status():
    """Check database status"""
    logger = logging.getLogger(__name__)
    
    logger.info("🗄️ Checking database status...")
    
    try:
        session = get_session()
        price_count = session.query(PriceData).count()
        sentiment_count = session.query(SentimentData).count()
        
        # Get latest price
        latest_price = session.query(PriceData).order_by(
            PriceData.timestamp.desc()).first()
        
        # Get latest sentiment
        latest_sentiment = session.query(SentimentData).order_by(
            SentimentData.timestamp.desc()).first()
        
        session.close()
        
        print("📊 Database Status:")
        print(f"  - Price records: {price_count}")
        print(f"  - Sentiment records: {sentiment_count}")
        latest_price_val = latest_price.price if latest_price else 'None'
        latest_price_time = latest_price.timestamp if latest_price else 'None'
        latest_sentiment_val = (latest_sentiment.sentiment_score
                                if latest_sentiment else 'None')
        latest_sentiment_time = (latest_sentiment.timestamp
                                 if latest_sentiment else 'None')
        print(f"  - Latest price: {latest_price_val}")
        print(f"  - Latest price time: {latest_price_time}")
        print(f"  - Latest sentiment: {latest_sentiment_val}")
        print(f"  - Latest sentiment time: {latest_sentiment_time}")
        
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    check_database_status()
    test_dashboard_apis()
