#!/usr/bin/env python3
"""
Test script for Bitcoin Analysis Bot
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test if all modules can be imported"""
    print("🧪 Testing imports...")
    
    try:
        from config.settings import settings
        print("✅ Config module imported successfully")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from database.models import create_database, get_session
        print("✅ Database module imported successfully")
    except Exception as e:
        print(f"❌ Database import failed: {e}")
        return False
    
    try:
        from data_collectors.price_collector import PriceCollector
        print("✅ Price collector imported successfully")
    except Exception as e:
        print(f"❌ Price collector import failed: {e}")
        return False
    
    try:
        from analysis.sentiment_analyzer import SentimentAnalyzer
        print("✅ Sentiment analyzer imported successfully")
    except Exception as e:
        print(f"❌ Sentiment analyzer import failed: {e}")
        return False
    
    try:
        from utils.helpers import format_price, health_check
        print("✅ Utils module imported successfully")
    except Exception as e:
        print(f"❌ Utils import failed: {e}")
        return False
    
    return True

def test_database():
    """Test database creation and connection"""
    print("\n🗄️ Testing database...")
    
    try:
        from database.models import create_database, get_session, PriceData
        
        # Create database
        create_database()
        print("✅ Database created successfully")
        
        # Test connection
        session = get_session()
        count = session.query(PriceData).count()
        session.close()
        print(f"✅ Database connection successful (found {count} price records)")
        
        return True
        
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_price_collection():
    """Test price data collection"""
    print("\n💰 Testing price collection...")
    
    try:
        from data_collectors.price_collector import PriceCollector
        
        collector = PriceCollector()
        
        # Test CoinGecko (no API key needed)
        price_data = collector.get_current_price_coingecko()
        
        if price_data and price_data.get('price'):
            print(f"✅ Price collection successful: ${price_data['price']:.2f}")
            return True
        else:
            print("⚠️ Price collection returned no data")
            return False
            
    except Exception as e:
        print(f"❌ Price collection test failed: {e}")
        return False

def test_sentiment_analysis():
    """Test sentiment analysis functionality"""
    print("\n💭 Testing sentiment analysis...")
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        from textblob import TextBlob
        
        # Test VADER
        analyzer = SentimentIntensityAnalyzer()
        test_text = "Bitcoin is going to the moon! This is amazing!"
        vader_score = analyzer.polarity_scores(test_text)
        print(f"✅ VADER sentiment analysis working: {vader_score['compound']:.3f}")
        
        # Test TextBlob
        blob = TextBlob(test_text)
        textblob_score = blob.sentiment.polarity
        print(f"✅ TextBlob sentiment analysis working: {textblob_score:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Sentiment analysis test failed: {e}")
        return False

def test_web_app():
    """Test web application"""
    print("\n🌐 Testing web application...")
    
    try:
        from web.app import create_app
        
        app = create_app()
        
        with app.test_client() as client:
            # Test main page
            response = client.get('/')
            if response.status_code == 200:
                print("✅ Web dashboard accessible")
            else:
                print(f"⚠️ Web dashboard returned status {response.status_code}")
            
            # Test API endpoint
            response = client.get('/api/stats')
            if response.status_code == 200:
                print("✅ API endpoints working")
                return True
            else:
                print(f"⚠️ API returned status {response.status_code}")
                return False
                
    except Exception as e:
        print(f"❌ Web application test failed: {e}")
        return False

def test_utilities():
    """Test utility functions"""
    print("\n🔧 Testing utilities...")
    
    try:
        from utils.helpers import format_price, format_percentage, health_check
        
        # Test formatting functions
        formatted_price = format_price(45678.123)
        formatted_pct = format_percentage(5.67)
        
        print(f"✅ Price formatting: {formatted_price}")
        print(f"✅ Percentage formatting: {formatted_pct}")
        
        # Test health check
        health = health_check()
        print(f"✅ Health check: Database={health['database']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Utilities test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Bitcoin Analysis Bot - Test Suite")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_database,
        test_price_collection,
        test_sentiment_analysis,
        test_web_app,
        test_utilities
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} crashed: {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Bot is ready to run.")
        return True
    elif passed >= total * 0.7:
        print("⚠️ Most tests passed. Bot should work but may have issues.")
        return True
    else:
        print("❌ Many tests failed. Please check your setup.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
