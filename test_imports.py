#!/usr/bin/env python3
"""
Test script to verify all imports and basic functionality
"""

import sys
import traceback

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    try:
        # Core libraries
        import logging
        import schedule
        import time
        import asyncio
        from datetime import datetime, timedelta
        print("✅ Core libraries imported")
        
        # Configuration
        from config.settings import settings
        print("✅ Settings imported")
        
        # Database
        from database.models import create_database
        print("✅ Database models imported")
        
        # Data collectors
        from data_collectors.twitter_collector import TwitterCollector
        from data_collectors.reddit_collector import RedditCollector
        from data_collectors.price_collector import PriceCollector
        print("✅ Data collectors imported")
        
        # Analyzers
        from analysis.sentiment_analyzer import SentimentAnalyzer
        from analysis.technical_analyzer import TechnicalAnalyzer
        from analysis.predictor import BitcoinPredictor
        print("✅ Analyzers imported")
        
        # Utilities
        from utils.helpers import send_telegram_message, create_alert_message
        print("✅ Utilities imported")
        
        # Main bot
        from main import BitcoinAnalysisBot
        print("✅ Main bot class imported")
        
        return True
        
    except Exception as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration values"""
    print("\n🔧 Testing configuration...")
    
    try:
        from config.settings import settings
        
        print(f"📊 Update interval: {settings.UPDATE_INTERVAL_MINUTES} minutes")
        print(f"🌐 Port: {settings.PORT}")
        print(f"📈 Prediction days: {settings.PREDICTION_DAYS}")
        print(f"💾 Database URL: {settings.DATABASE_URL}")
        
        # Check if critical settings are available
        has_telegram = bool(settings.TELEGRAM_BOT_TOKEN and settings.TELEGRAM_CHAT_ID)
        print(f"📱 Telegram configured: {has_telegram}")
        
        has_twitter = bool(settings.TWITTER_BEARER_TOKEN)
        print(f"🐦 Twitter configured: {has_twitter}")
        
        has_reddit = bool(settings.REDDIT_CLIENT_ID and settings.REDDIT_CLIENT_SECRET)
        print(f"📖 Reddit configured: {has_reddit}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return False

def test_database():
    """Test database connectivity"""
    print("\n💾 Testing database...")
    
    try:
        from database.models import create_database, get_session
        
        # Try to create database
        create_database()
        print("✅ Database created/connected")
        
        # Try to get session
        session = get_session()
        session.close()
        print("✅ Database session works")
        
        return True
        
    except Exception as e:
        print(f"❌ Database error: {e}")
        return False

def test_basic_functionality():
    """Test basic bot functionality"""
    print("\n🤖 Testing basic bot functionality...")
    
    try:
        from main import BitcoinAnalysisBot
        
        # Initialize bot (but don't start it)
        bot = BitcoinAnalysisBot()
        print("✅ Bot initialized")
        
        # Test individual components
        print("🔍 Testing price collector...")
        price_stats = bot.price_collector.collect_and_save()
        if price_stats:
            print("✅ Price collection works")
        else:
            print("⚠️ Price collection returned no data")
        
        return True
        
    except Exception as e:
        print(f"❌ Bot functionality error: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("🚀 Bitcoin Analysis Bot - Component Tests")
    print("=" * 50)
    
    all_passed = True
    
    # Run tests
    all_passed &= test_imports()
    all_passed &= test_configuration()
    all_passed &= test_database()
    all_passed &= test_basic_functionality()
    
    print("\n" + "=" * 50)
    if all_passed:
        print("✅ All tests passed! Bot should work correctly.")
    else:
        print("❌ Some tests failed. Check errors above.")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
