"""
Utility functions for the Bitcoin Analysis Bot
"""

import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
import json
import os
from config.settings import settings

def setup_logger(name, log_file='bitcoin_bot.log', level=logging.INFO):
    """Setup logger with file and console handlers"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Create formatters
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    console_formatter = logging.Formatter(
        '%(levelname)s - %(message)s'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    return logger

def send_telegram_message(message, chat_id=None, bot_token=None):
    """Send message via Telegram bot"""
    try:
        import requests
        
        bot_token = bot_token or settings.TELEGRAM_BOT_TOKEN
        chat_id = chat_id or settings.TELEGRAM_CHAT_ID
        
        if not bot_token or not chat_id:
            return False
        
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            'chat_id': chat_id,
            'text': message,
            'parse_mode': 'Markdown'
        }
        
        response = requests.post(url, data=data, timeout=10)
        return response.status_code == 200
        
    except Exception as e:
        logging.error(f"Failed to send Telegram message: {e}")
        return False

def format_price(price):
    """Format price with appropriate decimal places"""
    if price is None:
        return "N/A"
    
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.4f}"
    else:
        return f"${price:.6f}"

def format_percentage(value):
    """Format percentage with sign"""
    if value is None:
        return "N/A"
    
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.2f}%"

def format_sentiment_score(score):
    """Format sentiment score with description"""
    if score is None:
        return "N/A"
    
    if score > 0.3:
        desc = "Very Positive"
    elif score > 0.1:
        desc = "Positive"
    elif score > -0.1:
        desc = "Neutral"
    elif score > -0.3:
        desc = "Negative"
    else:
        desc = "Very Negative"
    
    return f"{score:.3f} ({desc})"

def create_alert_message(analysis_report):
    """Create formatted alert message"""
    try:
        message = "🔮 **Bitcoin Analysis Alert**\n\n"
        
        # Price info
        if analysis_report.get('price_data'):
            price = analysis_report['price_data'].get('current_price')
            if price:
                message += f"💰 **Current Price:** {format_price(price)}\n"
        
        # Sentiment info
        if analysis_report.get('sentiment'):
            sentiment = analysis_report['sentiment']['overall']
            message += f"💭 **Sentiment:** {format_sentiment_score(sentiment['overall_score'])}\n"
            message += f"📊 **Posts Analyzed:** {sentiment['total_posts']}\n"
        
        # Technical analysis
        if analysis_report.get('technical'):
            technical = analysis_report['technical']
            message += f"📈 **Technical:** {technical['recommendation']}\n"
        
        # Predictions
        if analysis_report.get('predictions') and analysis_report['predictions'].get('predictions'):
            predictions = analysis_report['predictions']['predictions']
            if '24h' in predictions:
                pred = predictions['24h']
                message += f"🔮 **24h Prediction:** {format_price(pred['predicted_price'])}\n"
                message += f"🎯 **Change:** {format_percentage(pred['price_change_pct'])}\n"
        
        message += f"\n⏰ **Time:** {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC"
        
        return message
        
    except Exception as e:
        logging.error(f"Error creating alert message: {e}")
        return "Error creating alert message"

def save_analysis_report(report, filename=None):
    """Save analysis report to JSON file"""
    try:
        if filename is None:
            timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
            filename = f"analysis_report_{timestamp}.json"
        
        # Create reports directory if it doesn't exist
        os.makedirs('reports', exist_ok=True)
        
        # Convert datetime objects to strings for JSON serialization
        def datetime_converter(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(os.path.join('reports', filename), 'w') as f:
            json.dump(report, f, indent=2, default=datetime_converter)
        
        return filename
        
    except Exception as e:
        logging.error(f"Error saving analysis report: {e}")
        return None

def load_analysis_report(filename):
    """Load analysis report from JSON file"""
    try:
        with open(os.path.join('reports', filename), 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading analysis report: {e}")
        return None

def calculate_accuracy(predictions, actual_prices):
    """Calculate prediction accuracy"""
    if not predictions or not actual_prices:
        return 0.0
    
    correct_predictions = 0
    total_predictions = 0
    
    for pred in predictions:
        if pred['prediction_id'] in actual_prices:
            predicted = pred['predicted_price']
            actual = actual_prices[pred['prediction_id']]
            
            # Consider prediction correct if within 5% of actual price
            error_percent = abs((predicted - actual) / actual) * 100
            if error_percent <= 5:
                correct_predictions += 1
            
            total_predictions += 1
    
    return (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0.0

def get_market_hours():
    """Check if market is in active hours (crypto markets are 24/7)"""
    return True  # Crypto markets never close

def is_significant_price_change(current_price, previous_price, threshold=5.0):
    """Check if price change is significant (default 5%)"""
    if not current_price or not previous_price:
        return False
    
    change_percent = abs((current_price - previous_price) / previous_price) * 100
    return change_percent >= threshold

def format_duration(seconds):
    """Format duration in seconds to human readable format"""
    if seconds < 60:
        return f"{seconds:.0f} seconds"
    elif seconds < 3600:
        return f"{seconds / 60:.1f} minutes"
    elif seconds < 86400:
        return f"{seconds / 3600:.1f} hours"
    else:
        return f"{seconds / 86400:.1f} days"

def clean_text_for_analysis(text):
    """Clean text for better sentiment analysis"""
    import re
    
    if not text:
        return ""
    
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    
    # Remove special characters but keep spaces
    text = re.sub(r'[^\w\s#@]', ' ', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def get_risk_level(volatility, sentiment_confidence, prediction_confidence):
    """Calculate overall risk level"""
    # Normalize inputs (assuming volatility is percentage)
    vol_score = min(volatility / 10, 1.0) if volatility else 0.5  # 10% vol = max risk
    sent_score = 1 - sentiment_confidence  # Lower confidence = higher risk
    pred_score = 1 - prediction_confidence  # Lower confidence = higher risk
    
    # Weighted average
    risk_score = (vol_score * 0.4 + sent_score * 0.3 + pred_score * 0.3)
    
    if risk_score > 0.7:
        return "HIGH"
    elif risk_score > 0.4:
        return "MEDIUM"
    else:
        return "LOW"

def create_backup(source_db, backup_dir='backups'):
    """Create database backup"""
    try:
        import shutil
        
        os.makedirs(backup_dir, exist_ok=True)
        
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        backup_name = f"bitcoin_analysis_backup_{timestamp}.db"
        backup_path = os.path.join(backup_dir, backup_name)
        
        shutil.copy2(source_db, backup_path)
        
        return backup_path
        
    except Exception as e:
        logging.error(f"Error creating backup: {e}")
        return None

def health_check():
    """Perform system health check"""
    health_status = {
        'timestamp': datetime.utcnow().isoformat(),
        'database': False,
        'api_keys': {},
        'disk_space': True,
        'memory': True
    }
    
    try:
        # Check database connection
        from database.models import get_session
        session = get_session()
        session.close()
        health_status['database'] = True
    except Exception:
        pass
    
    # Check API keys
    health_status['api_keys'] = {
        'twitter': bool(settings.TWITTER_BEARER_TOKEN),
        'reddit': bool(settings.REDDIT_CLIENT_ID),
        'binance': bool(settings.BINANCE_API_KEY),
        'coingecko': bool(settings.COINGECKO_API_KEY),
        'telegram': bool(settings.TELEGRAM_BOT_TOKEN)
    }
    
    return health_status

if __name__ == "__main__":
    # Test utility functions
    logger = setup_logger(__name__)
    logger.info("Testing utility functions...")
    
    print("Price formatting:", format_price(45678.123))
    print("Percentage formatting:", format_percentage(5.67))
    print("Sentiment formatting:", format_sentiment_score(0.234))
    
    health = health_check()
    print("Health check:", health)
