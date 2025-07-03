#!/usr/bin/env python3
"""
Тест для генерації тестового сигналу
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.helpers import send_telegram_message, create_alert_message
from datetime import datetime

def send_test_signal():
    """Відправити тестовий сигнал"""
    
    # Створити тестові дані з сигналом
    test_analysis = {
        'timestamp': datetime.utcnow(),
        'price_data': {
            'current_price': 109733.92,
            'price_change_24h': 3.2  # 3.2% зміна - має створити регулярний сигнал
        },
        'sentiment': {
            'overall': {
                'overall_score': 0.45,  # Помірно позитивний
                'confidence': 0.78,
                'total_posts': 42
            }
        },
        'technical': {
            'recommendation': 'BUY',  # Сигнал на покупку
            'signals': {
                'strength': 0.72
            }
        },
        'predictions': {
            'predictions': {
                '24h': {
                    'predicted_price': 113250.00,
                    'price_change_pct': 3.2,
                    'confidence': 0.81
                }
            }
        }
    }
    
    # Створити повідомлення
    message = create_alert_message(test_analysis, alert_type="UPDATE")
    
    print("🔮 Тестовий сигнал:")
    print(message)
    print("\n" + "="*50)
    
    # Відправити в Telegram
    try:
        success = send_telegram_message(message)
        if success:
            print("✅ Тестовий сигнал відправлено в Telegram!")
        else:
            print("❌ Помилка відправки в Telegram")
        return success
    except Exception as e:
        print(f"❌ Помилка: {e}")
        return False

if __name__ == "__main__":
    print("📱 Генерація тестового трейдинг сигналу...")
    send_test_signal()
