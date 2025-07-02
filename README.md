# 🔮 Bitcoin Analysis Bot

Багатофункціональний бот для аналізу та прогнозування ціни Bitcoin на основі:
- 💭 **Sentiment Analysis** - аналіз тональності постів з Twitter та Reddit
- 📈 **Technical Analysis** - технічний аналіз цінових графіків з 10+ індикаторами
- 🤖 **Machine Learning** - прогнозування ціни за допомогою RandomForest
- 🌐 **Web Dashboard** - красивий веб-інтерфейс з графіками
- � **24/7 Monitoring** - безперервний моніторинг ринку
- 📱 **Telegram Alerts** - сповіщення про важливі зміни

## ✨ Можливості
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

4. Запустіть бота:
```bash
python main.py
```

## API Ключі

Вам потрібно отримати наступні API ключі:

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
