#!/usr/bin/env python3
"""
Real-time Bitcoin Bot Monitor
Shows live updates from the running bot on the server
"""

import time
import os
import sqlite3
import pandas as pd
from datetime import datetime, timedelta
import requests
import json

def clear_screen():
    """Clear terminal screen"""
    os.system('cls' if os.name == 'nt' else 'clear')

def get_latest_data():
    """Get latest data from database"""
    try:
        conn = sqlite3.connect('bitcoin_sentiment.db')
        
        # Latest price
        price_query = """
            SELECT price, timestamp FROM price_data 
            ORDER BY timestamp DESC LIMIT 1
        """
        price_result = conn.execute(price_query).fetchone()
        
        # Recent sentiment (last hour)
        cutoff = datetime.now() - timedelta(hours=1)
        sentiment_query = """
            SELECT AVG(sentiment_score) as avg_sentiment, COUNT(*) as count
            FROM sentiment_data 
            WHERE timestamp > ?
        """
        sentiment_result = conn.execute(sentiment_query, (cutoff,)).fetchone()
        
        # Total records
        total_price = conn.execute("SELECT COUNT(*) FROM price_data").fetchone()[0]
        total_sentiment = conn.execute("SELECT COUNT(*) FROM sentiment_data").fetchone()[0]
        
        conn.close()
        
        return {
            'latest_price': price_result[0] if price_result else None,
            'price_time': price_result[1] if price_result else None,
            'avg_sentiment': sentiment_result[0] if sentiment_result[0] else 0,
            'recent_posts': sentiment_result[1] if sentiment_result else 0,
            'total_price_records': total_price,
            'total_sentiment_records': total_sentiment
        }
    except Exception as e:
        return {'error': str(e)}

def get_bot_status():
    """Check if bot process is running"""
    try:
        result = os.popen("pgrep -f 'python.*main.py'").read().strip()
        return bool(result)
    except:
        return False

def get_dashboard_status():
    """Check if dashboard is accessible"""
    try:
        response = requests.get('http://localhost:8000/api/price-data', timeout=5)
        return response.status_code == 200
    except:
        return False

def format_time_ago(timestamp_str):
    """Format time difference"""
    try:
        timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
        now = datetime.now()
        diff = now - timestamp
        
        if diff.seconds < 60:
            return f"{diff.seconds}s ago"
        elif diff.seconds < 3600:
            return f"{diff.seconds//60}m ago"
        else:
            return f"{diff.seconds//3600}h ago"
    except:
        return "unknown"

def display_status():
    """Display current bot status"""
    clear_screen()
    
    print("🤖 Bitcoin Sentiment Bot - Live Monitor")
    print("=" * 50)
    print(f"⏰ Last Update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Bot process status
    bot_running = get_bot_status()
    status_emoji = "🟢" if bot_running else "🔴"
    print(f"{status_emoji} Bot Process: {'Running' if bot_running else 'Stopped'}")
    
    # Dashboard status
    dashboard_running = get_dashboard_status()
    dash_emoji = "🟢" if dashboard_running else "🔴"
    print(f"{dash_emoji} Dashboard: {'Available' if dashboard_running else 'Unavailable'}")
    print()
    
    # Data status
    data = get_latest_data()
    
    if 'error' in data:
        print(f"❌ Database Error: {data['error']}")
    else:
        # Latest price
        if data['latest_price']:
            price_time = format_time_ago(data['price_time'])
            print(f"💰 Latest Price: ${data['latest_price']:,.2f} ({price_time})")
        else:
            print("💰 Latest Price: No data")
        
        # Sentiment
        sentiment = data['avg_sentiment']
        if sentiment > 0.1:
            sentiment_emoji = "😊"
        elif sentiment < -0.1:
            sentiment_emoji = "😔"
        else:
            sentiment_emoji = "😐"
        
        print(f"📊 Sentiment (1h): {sentiment_emoji} {sentiment:.3f} ({data['recent_posts']} posts)")
        
        # Data totals
        print(f"📈 Total Records: {data['total_price_records']} price, {data['total_sentiment_records']} sentiment")
    
    print()
    print("📱 Recent Activity:")
    
    # Check recent log entries
    try:
        if os.path.exists('bitcoin_bot.log'):
            with open('bitcoin_bot.log', 'r') as f:
                lines = f.readlines()
                recent_lines = lines[-5:] if len(lines) >= 5 else lines
                
                for line in recent_lines:
                    # Extract timestamp and message
                    if ' - ' in line:
                        parts = line.strip().split(' - ', 2)
                        if len(parts) >= 3:
                            timestamp_part = parts[0]
                            level = parts[1]
                            message = parts[2]
                            
                            # Color code by level
                            if 'ERROR' in level:
                                print(f"🔴 {timestamp_part[-8:]}: {message[:60]}...")
                            elif 'WARNING' in level:
                                print(f"🟡 {timestamp_part[-8:]}: {message[:60]}...")
                            elif 'INFO' in level:
                                print(f"🟢 {timestamp_part[-8:]}: {message[:60]}...")
                            else:
                                print(f"⚪ {timestamp_part[-8:]}: {message[:60]}...")
        else:
            print("⚪ No log file found")
    except Exception as e:
        print(f"⚪ Log read error: {e}")
    
    print()
    print("🔄 Auto-refreshing every 30 seconds...")
    print("💡 Press Ctrl+C to exit")

def main():
    """Main monitoring loop"""
    print("🚀 Starting Bitcoin Bot Monitor...")
    print("Monitoring bot activity in real-time...")
    
    try:
        while True:
            display_status()
            time.sleep(30)  # Update every 30 seconds
    except KeyboardInterrupt:
        print("\n👋 Monitor stopped by user")
    except Exception as e:
        print(f"\n❌ Monitor error: {e}")

if __name__ == "__main__":
    main()
