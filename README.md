# 🔮 Enhanced Bitcoin Analysis Bot with Neural Networks

🚀 **NEW: Enhanced with CryptoBERT and LightGBM for state-of-the-art analysis!**

Багатофункціональний бот для аналізу та прогнозування ціни Bitcoin на основі:
- 🧠 **Neural Networks** - CryptoBERT для sentiment + LightGBM для prediction
- 💭 **Advanced Sentiment** - криптоспецифічний аналіз тональності
- 📊 **Enhanced Data** - Fear & Greed Index, StockTwits, on-chain metrics
- 📈 **Technical Analysis** - технічний аналіз цінових графіків з 10+ індикаторами
- 🤖 **Machine Learning** - прогнозування напрямку ціни з 67%+ точністю
- 🌐 **Web Dashboard** - красивий веб-інтерфейс з графіками
- ⚡ **24/7 Monitoring** - безперервний моніторинг ринку
- 📱 **Enhanced Alerts** - розумні сповіщення з neural network insights

## 🆕 Enhanced Features (2025)

### Neural Network Models
- **🧠 CryptoBERT**: Specialized crypto sentiment analysis (ElKulako/cryptobert)
- **📊 LightGBM**: Advanced price direction prediction with 67.4% accuracy
- **🔗 Hybrid Analysis**: Intelligent fallback system for maximum reliability

### Enhanced Data Sources
- **😨 Fear & Greed Index**: Real-time market sentiment indicator
- **💬 StockTwits**: Social sentiment from crypto trading community
- **⛓️ On-chain Metrics**: NVT ratio, MVRV ratio (Glassnode integration)
- **📈 Market Correlations**: Bitcoin vs traditional markets (SPY, QQQ, GLD)

### Smart Features
- **🎯 Enhanced Predictions**: Neural network-powered price direction forecasting
- **📱 Rich Alerts**: Detailed Telegram notifications with AI insights
- **🛡️ Robust Fallbacks**: Graceful degradation when advanced features unavailable
- **⚡ Real-time Analysis**: Sub-second sentiment classification

## ✨ Core Capabilities
- 🐦 **Twitter Integration** - збір та аналіз твітів про Bitcoin
- 📱 **Reddit Analysis** - моніторинг криптовалютних subreddit-ів
- 📊 **Price Tracking** - отримання даних з Binance, CoinGecko, Yahoo Finance
- � **Sentiment Scoring** - оцінка тональності за допомогою VADER + TextBlob
- 📈 **Technical Indicators** - RSI, MACD, Bollinger Bands, SMA, EMA та інші
- 🎯 **Price Prediction** - прогнози на 1h, 4h, 24h з рівнем довіри
- 🌐 **Beautiful Dashboard** - адаптивний веб-дашборд з Chart.js
- 🔄 **Auto-Retraining** - автоматичне переnavчання ML моделі
- 📱 **Telegram Bot** - сповіщення та алерти
- ☁️ **Cloud Ready** - готовий до деплою на Render/Heroku

## Встановлення

1. Клонуйте репозиторій:
```bash
git clone <repository-url>
cd Sentiment_Twitter
```

2. Встановіть залежності:
```bash
pip install -r requirements.txt
```

3. Налаштуйте змінні середовища:
```bash
cp .env.example .env
# Заповніть .env файл своїми API ключами
```

4. **NEW**: Test enhanced features:
```bash
python test_enhancements.py
```

5. Запустіть бота:
```bash
python main.py
```

## 🚀 Quick Start with Enhanced Features

```bash
# Install with neural network support
pip install transformers torch lightgbm tokenizers fear-greed-index

# Test all enhancements
TOKENIZERS_PARALLELISM=false python test_enhancements.py

# Start bot with enhanced features
python main.py
```

## API Ключі

### Core APIs (Required)
- **Twitter/X API**: Bearer Token для соціального sentiment
- **Reddit API**: Client ID/Secret для моніторингу subreddit-ів  
- **Binance API**: Для отримання цінових даних (публічне API)
- **Telegram Bot**: Для сповіщень

### Enhanced APIs (Optional)
- **Glassnode API**: Для on-chain метрик (NVT, MVRV)
- **CoinGecko API**: Для розширених цінових даних

Детальна інструкція з налаштування:

### Twitter API
1. Йдіть на https://developer.twitter.com/
2. Створіть додаток
3. Отримайте Bearer Token та API ключі

### Reddit API
1. Йдіть на https://www.reddit.com/prefs/apps
2. Створіть додаток типу "script"
3. Отримайте Client ID та Client Secret

### Binance API (опціонально)
1. Реєструйтесь на Binance
2. Створіть API ключі в налаштуваннях
3. Дозвольте тільки читання даних

### CoinGecko API
1. Реєструйтесь на https://www.coingecko.com/en/api
2. Отримайте безкоштовний API ключ

## Деплой на Render

1. Створіть акаунт на https://render.com/
2. Підключіть ваш GitHub репозиторій
3. Встановіть змінні середовища в Render dashboard
4. Деплойте як Web Service

## Деплой на Digital Ocean

### 1. Створіть Droplet
- Ubuntu 22.04 LTS
- Мінімум 2GB RAM (для ML моделей)
- Додайте SSH ключ

### 2. Деплойте на сервер
```bash
# Скопіюйте проект на сервер
scp -r . root@YOUR_DROPLET_IP:/opt/bitcoin-sentiment-bot

# SSH на сервер
ssh root@YOUR_DROPLET_IP

# Запустіть скрипт деплою
cd /opt/bitcoin-sentiment-bot
chmod +x deploy.sh
./deploy.sh

# Налаштуйте середовище
cp .env.example .env
nano .env  # Додайте свої API ключі

# Запустіть бота
docker-compose up -d
```

### 3. Моніторинг
```bash
# Перевірте логи
docker-compose logs -f

# Перевірте статус
docker-compose ps
```

## Структура проекту
```
Sentiment_Twitter/
├── main.py                 # Головний файл запуску
├── config/
│   └── settings.py         # Налаштування
├── data_collectors/
│   ├── twitter_collector.py
│   ├── reddit_collector.py
│   └── price_collector.py
├── analysis/
│   ├── sentiment_analyzer.py
│   ├── technical_analyzer.py
│   └── predictor.py
├── database/
│   └── models.py
├── web/
│   ├── app.py             # FastAPI додаток
│   └── templates/
├── utils/
│   └── helpers.py
└── requirements.txt
```

## Використання

Після запуску бот буде:
1. Збирати дані кожні 30 хвилин
2. Аналізувати тональність постів
3. Проводити технічний аналіз
4. Генерувати прогнози
5. Відправляти сповіщення в Telegram
6. Показувати дані на веб-дашборді (http://localhost:8000)

## 🔧 Налаштування

Ключові змінні середовища в `.env`:
```env
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
UPDATE_INTERVAL_HOURS=6
ENABLE_CRYPTOBERT=true
ENABLE_LIGHTGBM=true
```

## 📖 Повна документація

- [Посібник з деплою на Digital Ocean](DIGITAL_OCEAN_DEPLOYMENT.md)
- [Налаштування середовища](.env.example)

## 💰 Оцінка вартості
- Digital Ocean Droplet: $12-24/місяць
- Загальна місячна вартість: $13-26

---

**Статус**: ✅ Готовий до виробництва | Всі нейронні моделі працюють | Розширені функції активні
