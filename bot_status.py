#!/usr/bin/env python3
"""
Скрипт моніторингу статусу Bitcoin Analysis Bot
"""

import subprocess
import os
import json
from datetime import datetime, timedelta
import requests

def check_bot_process():
    """Перевірити, чи запущений процес бота"""
    try:
        result = subprocess.run(
            ["pgrep", "-f", "python.*main.py"], 
            capture_output=True, 
            text=True
        )
        
        if result.returncode == 0:
            pids = result.stdout.strip().split('\n')
            pids = [pid for pid in pids if pid]  # Видалити порожні
            return pids
        else:
            return []
            
    except Exception as e:
        print(f"❌ Помилка перевірки процесу: {e}")
        return []

def check_web_server():
    """Перевірити, чи працює веб-сервер"""
    try:
        response = requests.get("http://localhost:8000", timeout=5)
        return response.status_code == 200
    except:
        return False

def check_recent_logs():
    """Перевірити останні записи в логах"""
    log_files = ["bot_clean.log", "bot_output_new.log", "bot_output.log", "bitcoin_bot.log"]
    
    for log_file in log_files:
        if os.path.exists(log_file):
            try:
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    
                if lines:
                    last_line = lines[-1].strip()
                    if last_line:
                        return log_file, last_line
                        
            except Exception as e:
                continue
                
    return None, None

def check_database_data():
    """Перевірити останні дані в базі"""
    try:
        import sys
        sys.path.append(os.path.dirname(os.path.abspath(__file__)))
        
        from database.models import get_session, SentimentData, PriceData
        
        session = get_session()
        
        # Перевірити останні sentiment дані
        cutoff = datetime.utcnow() - timedelta(hours=1)
        recent_sentiment = session.query(SentimentData).filter(
            SentimentData.timestamp >= cutoff
        ).count()
        
        # Перевірити останні price дані
        recent_price = session.query(PriceData).filter(
            PriceData.timestamp >= cutoff
        ).count()
        
        # Остання ціна
        latest_price = session.query(PriceData).order_by(
            PriceData.timestamp.desc()
        ).first()
        
        session.close()
        
        return {
            'sentiment_entries': recent_sentiment,
            'price_entries': recent_price,
            'latest_price': latest_price.price if latest_price else None,
            'latest_price_time': latest_price.timestamp if latest_price else None
        }
        
    except Exception as e:
        return {'error': str(e)}

def main():
    """Головна функція моніторингу"""
    print("🔍 Bitcoin Analysis Bot - Статус моніторинг")
    print("=" * 50)
    
    # Перевірити процес
    pids = check_bot_process()
    if pids:
        print(f"✅ Бот запущений (PID: {', '.join(pids)})")
        if len(pids) > 1:
            print("⚠️  УВАГА: Запущено кілька процесів!")
    else:
        print("❌ Бот НЕ запущений")
    
    # Перевірити веб-сервер
    web_status = check_web_server()
    if web_status:
        print("✅ Веб-дашборд доступний: http://localhost:8000")
    else:
        print("❌ Веб-дашборд недоступний")
    
    # Перевірити логи
    log_file, last_log = check_recent_logs()
    if last_log:
        print(f"📝 Останній лог ({log_file}):")
        print(f"   {last_log}")
    else:
        print("❌ Немає доступних логів")
    
    # Перевірити дані в базі
    print("\n📊 Дані в базі (остання година):")
    db_data = check_database_data()
    
    if 'error' in db_data:
        print(f"❌ Помилка бази даних: {db_data['error']}")
    else:
        print(f"   💭 Sentiment записів: {db_data['sentiment_entries']}")
        print(f"   💰 Price записів: {db_data['price_entries']}")
        if db_data['latest_price']:
            print(f"   📈 Остання ціна: ${db_data['latest_price']:,.2f}")
            print(f"   ⏰ Час: {db_data['latest_price_time']}")
    
    # Статус висновок
    print("\n" + "=" * 50)
    
    if pids and web_status and db_data.get('sentiment_entries', 0) > 0:
        print("🎉 СТАТУС: Бот працює нормально!")
    elif pids and web_status:
        print("⚠️  СТАТУС: Бот працює, але мало даних")
    elif pids:
        print("⚠️  СТАТУС: Бот працює, але є проблеми")
    else:
        print("❌ СТАТУС: Бот не працює!")
    
    print("\n🔧 Команди:")
    print("   Запустити: python main.py > bot.log 2>&1 &")
    print("   Зупинити: pkill -f 'python main.py'")
    print("   Дашборд: http://localhost:8000")
    print("   Логи: tail -f bot_clean.log")

if __name__ == "__main__":
    main()
