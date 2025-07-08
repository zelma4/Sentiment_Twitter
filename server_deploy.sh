#!/bin/bash
# 
# Bitcoin Bot Server Deployment & Management Script
# Optimized for production server deployment
#

set -e  # Exit on any error

BOT_DIR="/app"  # Adjust to your server path
PYTHON_CMD="python3"
LOG_FILE="bitcoin_bot.log"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_blue() {
    echo -e "${BLUE}[DEPLOY]${NC} $1"
}

# Check if running in correct directory
check_directory() {
    if [[ ! -f "main.py" ]]; then
        log_error "main.py not found. Please run this script from the bot directory."
        exit 1
    fi
    log_info "✅ Bot directory confirmed"
}

# Check system requirements
check_requirements() {
    log_blue "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 not found"
        exit 1
    fi
    log_info "✅ Python3: $(python3 --version)"
    
    # Check pip
    if ! command -v pip3 &> /dev/null; then
        log_error "pip3 not found"
        exit 1
    fi
    log_info "✅ pip3 available"
    
    # Check disk space (need at least 1GB)
    AVAILABLE_SPACE=$(df . | tail -1 | awk '{print $4}')
    if [[ $AVAILABLE_SPACE -lt 1048576 ]]; then
        log_warn "⚠️ Low disk space: $(df -h . | tail -1 | awk '{print $4}') available"
    else
        log_info "✅ Disk space: $(df -h . | tail -1 | awk '{print $4}') available"
    fi
}

# Install/update Python packages
install_packages() {
    log_blue "Installing/updating Python packages..."
    
    # Check if requirements.txt exists
    if [[ -f "requirements.txt" ]]; then
        log_info "Installing from requirements.txt..."
        pip3 install -r requirements.txt --upgrade
    else
        log_warn "requirements.txt not found, installing core packages..."
        pip3 install pandas numpy scikit-learn lightgbm transformers torch torchvision \
                    requests tweepy praw sqlite3 schedule python-telegram-bot matplotlib \
                    seaborn plotly flask flask-cors ccxt python-dotenv pytrends
    fi
    
    log_info "✅ Packages installed/updated"
}

# Setup database and directories
setup_environment() {
    log_blue "Setting up environment..."
    
    # Create necessary directories
    mkdir -p models logs data
    mkdir -p models/lightgbm models/advanced models/multi_horizon
    
    # Set permissions
    chmod 755 models logs data
    
    # Initialize database if needed
    if [[ ! -f "bitcoin_sentiment.db" ]]; then
        log_info "Initializing database..."
        $PYTHON_CMD -c "
from database.models import create_database
create_database()
print('Database initialized')
"
    fi
    
    log_info "✅ Environment setup complete"
}

# Check configuration
check_config() {
    log_blue "Checking configuration..."
    
    # Check if .env file exists
    if [[ ! -f ".env" ]]; then
        log_warn "⚠️ .env file not found"
        log_info "Creating sample .env file..."
        cat > .env << EOF
# Twitter API (optional)
TWITTER_BEARER_TOKEN=your_bearer_token_here

# Reddit API
REDDIT_CLIENT_ID=your_client_id
REDDIT_CLIENT_SECRET=your_client_secret
REDDIT_USER_AGENT=bitcoin_bot_v1.0

# Telegram Bot (for alerts)
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# API Keys (optional)
COINGECKO_API_KEY=your_api_key
GLASSNODE_API_KEY=your_api_key

# Bot Settings
UPDATE_INTERVAL_MINUTES=10
SYMBOL=BTC
CRYPTO_NAME=Bitcoin
PREDICTION_DAYS=3
EOF
        log_warn "⚠️ Please edit .env file with your API keys"
        return 1
    else
        log_info "✅ .env file found"
        
        # Check for essential keys
        if grep -q "REDDIT_CLIENT_ID=your_client_id" .env; then
            log_warn "⚠️ Reddit API not configured in .env"
        fi
        
        if grep -q "TELEGRAM_BOT_TOKEN=your_bot_token" .env; then
            log_warn "⚠️ Telegram bot not configured in .env"
        fi
    fi
    
    return 0
}

# Stop existing bot
stop_bot() {
    log_blue "Stopping existing bot processes..."
    
    # Find and kill bot processes
    BOT_PIDS=$(pgrep -f "python.*main.py" || true)
    
    if [[ -n "$BOT_PIDS" ]]; then
        log_info "Stopping bot processes: $BOT_PIDS"
        echo "$BOT_PIDS" | xargs kill -TERM
        sleep 5
        
        # Force kill if still running
        BOT_PIDS=$(pgrep -f "python.*main.py" || true)
        if [[ -n "$BOT_PIDS" ]]; then
            log_warn "Force killing bot processes: $BOT_PIDS"
            echo "$BOT_PIDS" | xargs kill -KILL
        fi
        
        log_info "✅ Bot processes stopped"
    else
        log_info "No bot processes running"
    fi
}

# Start bot
start_bot() {
    log_blue "Starting Bitcoin bot..."
    
    # Start bot in background
    nohup $PYTHON_CMD main.py > $LOG_FILE 2>&1 &
    BOT_PID=$!
    
    log_info "✅ Bot started with PID: $BOT_PID"
    
    # Wait a moment and check if it's still running
    sleep 3
    if kill -0 $BOT_PID 2>/dev/null; then
        log_info "✅ Bot is running successfully"
        
        # Show recent logs
        log_info "Recent log output:"
        tail -10 $LOG_FILE
        
        return 0
    else
        log_error "❌ Bot failed to start"
        log_info "Recent log output:"
        tail -20 $LOG_FILE
        return 1
    fi
}

# Show status
show_status() {
    log_blue "Bot Status:"
    
    # Check if bot is running
    if pgrep -f "python.*main.py" > /dev/null; then
        BOT_PID=$(pgrep -f "python.*main.py")
        log_info "✅ Bot is running (PID: $BOT_PID)"
        
        # Show memory usage
        MEMORY=$(ps -p $BOT_PID -o rss= 2>/dev/null || echo "unknown")
        if [[ "$MEMORY" != "unknown" ]]; then
            MEMORY_MB=$((MEMORY / 1024))
            log_info "💾 Memory usage: ${MEMORY_MB}MB"
        fi
    else
        log_warn "❌ Bot is not running"
    fi
    
    # Check log file
    if [[ -f "$LOG_FILE" ]]; then
        LOG_SIZE=$(du -h $LOG_FILE | cut -f1)
        log_info "📄 Log file: $LOG_FILE ($LOG_SIZE)"
        
        # Show recent activity
        log_info "📊 Recent activity:"
        tail -5 $LOG_FILE | while read line; do
            echo "   $line"
        done
    else
        log_warn "📄 No log file found"
    fi
    
    # Check database
    if [[ -f "bitcoin_sentiment.db" ]]; then
        DB_SIZE=$(du -h bitcoin_sentiment.db | cut -f1)
        log_info "💾 Database: bitcoin_sentiment.db ($DB_SIZE)"
        
        # Get record counts
        PRICE_COUNT=$($PYTHON_CMD -c "
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
        
        SENTIMENT_COUNT=$($PYTHON_CMD -c "
import sqlite3
try:
    conn = sqlite3.connect('bitcoin_sentiment.db')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM sentiment_data')
    print(cursor.fetchone()[0])
    conn.close()
except:
    print(0)
" 2>/dev/null)
        
        log_info "📈 Data: $PRICE_COUNT price records, $SENTIMENT_COUNT sentiment records"
    else
        log_warn "💾 No database found"
    fi
}

# Train models
train_models() {
    log_blue "Training models with current data..."
    
    if [[ -f "train_server_models.py" ]]; then
        $PYTHON_CMD train_server_models.py
    else
        log_warn "train_server_models.py not found, using quick_init_models.py"
        if [[ -f "quick_init_models.py" ]]; then
            $PYTHON_CMD quick_init_models.py
        else
            log_error "No training script found"
            return 1
        fi
    fi
}

# Show help
show_help() {
    cat << EOF
Bitcoin Bot Server Management Script

Usage: $0 [COMMAND]

Commands:
    deploy      - Full deployment (install, setup, start)
    start       - Start the bot
    stop        - Stop the bot  
    restart     - Restart the bot
    status      - Show bot status
    logs        - Show recent logs
    train       - Train models with current data
    monitor     - Start real-time monitor
    install     - Install/update packages only
    setup       - Setup environment only
    help        - Show this help

Examples:
    $0 deploy       # Full deployment
    $0 restart      # Restart bot
    $0 status       # Check status
    $0 logs         # View recent logs

EOF
}

# Main script logic
case "${1:-help}" in
    "deploy")
        log_blue "🚀 Starting full deployment..."
        check_directory
        check_requirements
        install_packages
        setup_environment
        if check_config; then
            stop_bot
            start_bot
            show_status
            log_blue "🎉 Deployment complete!"
            log_info "Monitor with: $0 monitor"
            log_info "Check status: $0 status"
        else
            log_error "Configuration incomplete. Please edit .env file and run again."
            exit 1
        fi
        ;;
    "start")
        check_directory
        start_bot
        ;;
    "stop")
        stop_bot
        ;;
    "restart")
        check_directory
        stop_bot
        sleep 2
        start_bot
        ;;
    "status")
        check_directory
        show_status
        ;;
    "logs")
        if [[ -f "$LOG_FILE" ]]; then
            tail -50 $LOG_FILE
        else
            log_error "Log file not found"
        fi
        ;;
    "train")
        check_directory
        train_models
        ;;
    "monitor")
        if [[ -f "monitor_bot.py" ]]; then
            $PYTHON_CMD monitor_bot.py
        else
            log_error "monitor_bot.py not found"
        fi
        ;;
    "install")
        check_directory
        install_packages
        ;;
    "setup")
        check_directory
        setup_environment
        ;;
    "help"|*)
        show_help
        ;;
esac
