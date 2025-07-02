#!/usr/bin/env python3
"""
Quick Start Script for Bitcoin Analysis Bot
"""

import os
import sys

def create_minimal_env():
    """Create a minimal .env file for testing"""
    env_content = """# Bitcoin Analysis Bot Environment Variables

# ==============================================
# ОБОВ'ЯЗКОВІ НАЛАШТУВАННЯ (заповніть ваші ключі)
# ==============================================

# Twitter API (required for sentiment analysis)
TWITTER_BEARER_TOKEN=your_twitter_bearer_token_here
TWITTER_API_KEY=your_twitter_api_key_here
TWITTER_API_SECRET=your_twitter_api_secret_here
TWITTER_ACCESS_TOKEN=your_twitter_access_token_here
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret_here

# Reddit API (required for sentiment analysis)
REDDIT_CLIENT_ID=your_reddit_client_id_here
REDDIT_CLIENT_SECRET=your_reddit_client_secret_here
REDDIT_USER_AGENT=BitcoinAnalysisBot/1.0

# ==============================================
# ОПЦІОНАЛЬНІ НАЛАШТУВАННЯ
# ==============================================

# Binance API (for better price data)
BINANCE_API_KEY=
BINANCE_SECRET_KEY=

# CoinGecko API (free alternative)
COINGECKO_API_KEY=

# Telegram Bot (for notifications)
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=

# ==============================================
# СИСТЕМНІ НАЛАШТУВАННЯ
# ==============================================

DATABASE_URL=sqlite:///bitcoin_analysis.db
UPDATE_INTERVAL_MINUTES=30
PREDICTION_DAYS=7
MIN_SENTIMENT_POSTS=20
PORT=8000
HOST=0.0.0.0
"""
    
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print("✅ Created .env file")
    print("⚠️  ВАЖЛИВО: Відредагуйте .env файл та додайте ваші API ключі!")

def main():
    print("🚀 Bitcoin Analysis Bot - Quick Start")
    print("=" * 50)
    
    # Check if .env exists
    if not os.path.exists('.env'):
        create_minimal_env()
    
    print("\n📋 Що потрібно зробити далі:")
    print("\n1. 🔑 ОТРИМАЙТЕ API КЛЮЧІ:")
    print("   • Twitter: https://developer.twitter.com/")
    print("   • Reddit: https://www.reddit.com/prefs/apps")
    print("   • (Опціонально) Binance, CoinGecko, Telegram")
    
    print("\n2. ⚙️  НАЛАШТУЙТЕ .env ФАЙЛ:")
    print("   • Відкрийте .env файл у текстовому редакторі")
    print("   • Замініть 'your_*_here' на ваші справжні API ключі")
    
    print("\n3. 📦 ВСТАНОВІТЬ ЗАЛЕЖНОСТІ:")
    print("   pip install -r requirements.txt")
    
    print("\n4. 🗄️  ІНІЦІАЛІЗУЙТЕ БАЗУ ДАНИХ:")
    print("   python -c \"from database.models import create_database; create_database()\"")
    
    print("\n5. 🧪 ПРОТЕСТУЙТЕ СИСТЕМУ:")
    print("   python test_bot.py")
    
    print("\n6. 🚀 ЗАПУСТІТЬ БОТА:")
    print("   python main.py")
    
    print("\n7. 🌐 ВІДКРИЙТЕ ДАШБОРД:")
    print("   http://localhost:8000")
    
    print("\n" + "=" * 50)
    print("📚 Детальні інструкції: читайте DEPLOYMENT.md")
    print("🐛 Проблеми? Запустіть: python test_bot.py")
    print("💡 Підказка: почніть з Twitter + Reddit API ключів")

if __name__ == "__main__":
    main()
