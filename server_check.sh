#!/bin/bash
"""
Server Bot Optimization Script
Runs advanced model training and improvements on the server
"""

echo "🚀 Starting Bitcoin Bot Server Optimization..."
echo "================================================"

# Navigate to bot directory
cd /home/ubuntu/bitcoin_sentiment_bot || {
    echo "❌ Bot directory not found"
    exit 1
}

echo "📊 Current directory: $(pwd)"

# Check if bot is running
if pgrep -f "python.*main.py" > /dev/null; then
    echo "✅ Bot is currently running"
    BOT_PID=$(pgrep -f "python.*main.py")
    echo "🔧 Bot PID: $BOT_PID"
else
    echo "❌ Bot is not running"
fi

# Check Python environment
echo ""
echo "🐍 Python Environment Check:"
python3 --version
pip3 list | grep -E "(torch|transformers|lightgbm|scikit-learn)" | head -10

# Check available models
echo ""
echo "📁 Available Models:"
if [ -d "models" ]; then
    ls -la models/
    find models/ -name "*.pth" -o -name "*.pkl" -o -name "*.joblib" 2>/dev/null | head -5
else
    echo "❌ Models directory not found"
fi

# Check recent logs
echo ""
echo "📝 Recent Bot Activity (last 20 lines):"
if [ -f "bitcoin_bot.log" ]; then
    tail -20 bitcoin_bot.log
else
    echo "❌ Log file not found"
fi

# Check database
echo ""
echo "💾 Database Status:"
if [ -f "bitcoin_sentiment.db" ]; then
    echo "✅ Database exists"
    echo "📊 Database size: $(du -h bitcoin_sentiment.db | cut -f1)"
    # Get record counts if possible
    python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('bitcoin_sentiment.db')
    cursor = conn.cursor()
    
    # Check tables
    cursor.execute('SELECT name FROM sqlite_master WHERE type=\"table\"')
    tables = cursor.fetchall()
    print(f'📋 Tables: {[t[0] for t in tables]}')
    
    # Count records in main tables
    for table in ['price_data', 'sentiment_data']:
        try:
            cursor.execute(f'SELECT COUNT(*) FROM {table}')
            count = cursor.fetchone()[0]
            print(f'📈 {table}: {count} records')
        except:
            pass
    
    conn.close()
except Exception as e:
    print(f'❌ Database check failed: {e}')
"
else
    echo "❌ Database not found"
fi

# Check disk space
echo ""
echo "💿 Server Resources:"
df -h . | head -2
echo "🧠 Memory usage:"
free -h | head -2

# Check if we can improve the bot
echo ""
echo "🔧 Optimization Opportunities:"

# Check if advanced models are trained
if [ ! -f "models/advanced_crypto_model.pth" ]; then
    echo "⚠️  Advanced Neural Network not trained"
    echo "   Run: python3 quick_init_models.py"
fi

if [ ! -f "models/lightgbm_model.pkl" ]; then
    echo "⚠️  LightGBM model not saved"
    echo "   Model will retrain automatically"
fi

# Check for sufficient historical data
PRICE_COUNT=$(python3 -c "
import sqlite3
try:
    conn = sqlite3.connect('bitcoin_sentiment.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM price_data')
    print(cursor.fetchone()[0])
    conn.close()
except:
    print(0)
" 2>/dev/null)

if [ "$PRICE_COUNT" -lt 100 ]; then
    echo "⚠️  Limited historical price data: $PRICE_COUNT records"
    echo "   Models will improve as data accumulates"
fi

echo ""
echo "🎯 Next Steps:"
echo "1. 📊 Monitor logs: tail -f bitcoin_bot.log"
echo "2. 🌐 Check dashboard: http://your-server:8000"
echo "3. 📱 Verify Telegram alerts are working"
echo "4. 🚀 Advanced models will train automatically on Sundays"

echo ""
echo "✅ Server optimization check complete!"
