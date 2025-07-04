#!/usr/bin/env python3
"""
Script to collect historical data for better dashboard experience.
This script should be run once to populate the database with historical data.
"""

import sys
import os
import logging

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_collectors.price_collector import PriceCollector
from data_collectors.reddit_collector import RedditCollector
from database.models import get_session, PriceData, SentimentData


def collect_historical_data():
    """Collect historical data to populate the database"""
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    logger.info("🔄 Starting historical data collection...")
    
    # Initialize collectors
    price_collector = PriceCollector()
    reddit_collector = RedditCollector()
    
    # Check current data
    session = get_session()
    price_count = session.query(PriceData).count()
    sentiment_count = session.query(SentimentData).count()
    session.close()
    
    logger.info(f"📊 Current data: {price_count} price records, "
                f"{sentiment_count} sentiment records")
    
    # Collect historical price data (last 7 days)
    if price_count < 10:
        logger.info("💰 Collecting historical price data...")
        try:
            price_stats = price_collector.collect_and_save(
                include_historical=True,
                historical_days=7
            )
            logger.info(f"✅ Price data collected: {price_stats}")
        except Exception as e:
            logger.error(f"❌ Price collection failed: {e}")
    
    # Collect some Reddit data if we don't have much
    if sentiment_count < 50:
        logger.info("💬 Collecting Reddit sentiment data...")
        try:
            reddit_stats = reddit_collector.collect_and_save(
                posts_limit=50,
                comments_limit=20
            )
            logger.info(f"✅ Reddit data collected: {reddit_stats}")
        except Exception as e:
            logger.error(f"❌ Reddit collection failed: {e}")
    
    # Final check
    session = get_session()
    final_price_count = session.query(PriceData).count()
    final_sentiment_count = session.query(SentimentData).count()
    session.close()
    
    logger.info(f"📈 Final data: {final_price_count} price records, "
                f"{final_sentiment_count} sentiment records")
    logger.info("✅ Historical data collection complete!")


if __name__ == "__main__":
    collect_historical_data()
