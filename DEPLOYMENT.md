# Bitcoin Analysis Bot - Deployment Guide

## � Варіанти запуску на Render

### Варіант 1: Тільки веб-дашборд (рекомендовано)
**Start Command:** `gunicorn --bind 0.0.0.0:$PORT wsgi:app`
- Запускає тільки Flask веб-сервер
- Найбільш стабільний варіант для безкоштовного хостингу
- Підходить, якщо потрібен тільки REST API та дашборд

### Варіант 2: Повний функціонал бота
**Start Command:** `python render_runner.py`
- Запускає і веб-сервер, і аналітичний бот
- Включає 24/7 моніторинг та збір даних
- Може бути обмежений безкоштовним планом Render

### Варіант 3: Тільки аналітичний бот (без веб-інтерфейсу)
**Start Command:** `python main.py`
- Запускає тільки бот для збору та аналізу даних
- Без веб-інтерфейсу (тільки логи)
- Підходить для фонових задач

## �📋 Перелік API ключів, які потрібно отримати

### 1. Twitter API (обов'язково для sentiment analysis)
- Йдіть на: https://developer.twitter.com/
- Створіть Developer Account
- Створіть новий App
- Отримайте:
  - `TWITTER_BEARER_TOKEN`
  - `TWITTER_API_KEY`
  - `TWITTER_API_SECRET`
  - `TWITTER_ACCESS_TOKEN`
  - `TWITTER_ACCESS_TOKEN_SECRET`

### 2. Reddit API (обов'язково для sentiment analysis)
- Йдіть на: https://www.reddit.com/prefs/apps
- Створіть додаток типу "script"
- Отримайте:
  - `REDDIT_CLIENT_ID`
  - `REDDIT_CLIENT_SECRET`
  - `REDDIT_USER_AGENT` (можна вказати "BitcoinAnalysisBot/1.0")

### 3. Binance API (опціонально, для кращих цінових даних)
- Зареєструйтесь на Binance
- Йдіть в API Management
- Створіть API ключі з правами тільки на читання
- Отримайте:
  - `BINANCE_API_KEY`
  - `BINANCE_SECRET_KEY`

### 4. CoinGecko API (опціонально, безкоштовна альтернатива)
- Зареєструйтесь на: https://www.coingecko.com/en/api
- Отримайте безкоштовний API ключ:
  - `COINGECKO_API_KEY`

### 5. Telegram Bot (опціонально, для сповіщень)
- Напишіть @BotFather в Telegram
- Створіть нового бота командою `/newbot`
- Отримайте:
  - `TELEGRAM_BOT_TOKEN`
  - `TELEGRAM_CHAT_ID` (ваш chat ID)

## 🚀 Локальний запуск

### 1. Клонування та підготовка
```bash
git clone <your-repo-url>
cd Sentiment_Twitter
```

### 2. Запуск setup скрипта
```bash
python setup.py
```

### 3. Редагування .env файла
```bash
nano .env
# Вставте ваші API ключі
```

### 4. Запуск бота
```bash
python main.py
```

### 5. Відкрийте дашборд
Перейдіть на: http://localhost:8000

## ☁️ Деплой на Render

### 1. Підготовка коду
```bash
git add .
git commit -m "Initial Bitcoin Analysis Bot"
git push origin main
```

### 2. Створення веб-сервісу на Render
1. Йдіть на https://render.com/
2. Натисніть "New" → "Web Service"
3. Підключіть ваш GitHub репозиторій
4. Налаштування:
   - **Name**: bitcoin-analysis-bot
   - **Environment**: Python 3
   - **Build Command**: `./build.sh`
   - **Start Command**: `gunicorn --bind 0.0.0.0:$PORT wsgi:app` (для веб-дашборду)
   
   **Альтернативні Start Commands:**
   - `python render_runner.py` (повний функціонал: бот + веб)
   - `python main.py` (тільки аналітичний бот)

### 3. Налаштування змінних середовища
В Render dashboard додайте Environment Variables:

```
TWITTER_BEARER_TOKEN=your_twitter_bearer_token
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_TOKEN_SECRET=your_twitter_access_token_secret

REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_client_secret
REDDIT_USER_AGENT=BitcoinAnalysisBot/1.0

BINANCE_API_KEY=your_binance_api_key (опціонально)
BINANCE_SECRET_KEY=your_binance_secret_key (опціонально)

COINGECKO_API_KEY=your_coingecko_api_key (опціонально)

TELEGRAM_BOT_TOKEN=your_telegram_bot_token (опціонально)
TELEGRAM_CHAT_ID=your_telegram_chat_id (опціонально)

DATABASE_URL=sqlite:///bitcoin_analysis.db
UPDATE_INTERVAL_MINUTES=30
PREDICTION_DAYS=7
MIN_SENTIMENT_POSTS=50
PORT=8000
```

### 4. Деплой
Натисніть "Deploy" - Render автоматично побудує та запустить ваш додаток.

## 📊 Як працює бот

### Збір даних (кожні 30 хвилин):
1. **Twitter** - збирає твіти про Bitcoin
2. **Reddit** - збирає пости з криптовалютних subreddit-ів
3. **Ціни** - отримує поточні ціни Bitcoin

### Аналіз:
1. **Sentiment Analysis** - аналізує тональність постів
2. **Technical Analysis** - розраховує технічні індикатори
3. **Price Prediction** - прогнозує ціну за допомогою ML

### Відображення:
1. **Web Dashboard** - показує всі дані та графіки
2. **API Endpoints** - надає дані в JSON форматі
3. **Telegram** - відправляє сповіщення (якщо налаштовано)

## 🔧 Налаштування

### Інтервал оновлення
Змініть `UPDATE_INTERVAL_MINUTES` в .env файлі (за замовчуванням 30 хвилин)

### Мінімальна кількість постів для аналізу
Змініть `MIN_SENTIMENT_POSTS` в .env файлі (за замовчуванням 50)

### Період прогнозування
Змініть `PREDICTION_DAYS` в .env файлі (за замовчуванням 7 днів)

## 📈 API Endpoints

- `GET /` - Головна сторінка дашборду
- `GET /api/stats` - Загальна статистика
- `GET /api/price-data` - Цінові дані за 24 години
- `GET /api/sentiment-summary` - Аналіз тональності
- `GET /api/technical-analysis` - Технічний аналіз
- `GET /api/predictions` - Прогнози цін

## 🐛 Вирішення проблем

### Помилка "No API keys"
- Переконайтесь, що ви заповнили .env файл
- Перевірте правильність API ключів

### Помилка "Database not found"
- Запустіть: `python -c "from database.models import create_database; create_database()"`

### Помилка "No data collected"
- Перевірте інтернет з'єднання
- Перевірте правильність API ключів Twitter/Reddit

### Низька точність прогнозів
- Дайте боту працювати кілька днів для збору даних
- Переконайтесь, що збирається достатньо даних

## 📞 Підтримка

Якщо у вас виникли питання або проблеми:
1. Перевірте логи: `tail -f bitcoin_bot.log`
2. Запустіть health check: `python -c "from utils.helpers import health_check; print(health_check())"`
3. Перевірте статус API: відкрийте `/api/stats` в браузері

## 🔄 Оновлення

Для оновлення коду:
```bash
git pull origin main
# Якщо є нові залежності:
pip install -r requirements.txt
# Перезапустіть бота
```

На Render оновлення відбуватиметься автоматично при push в main branch.

## 💡 Поради

1. **Починайте з мінімальних API ключів** - Twitter та Reddit достатньо для початку
2. **Дайте часу на збір даних** - перші прогнози можуть бути неточними
3. **Моніторьте логи** - вони показують всю активність бота
4. **Регулярно робіть backup** бази даних (функція вбудована)
5. **Налаштуйте Telegram** для отримання сповіщень

## 🚀 ГОТОВНІСТЬ ДО ДЕПЛОЮ

### ✅ Що готово:
1. **Додано gunicorn** до requirements.txt
2. **Створено wsgi.py** - entry point для web service  
3. **Створено render_runner.py** - для повного функціоналу
4. **Налаштовано Procfile** з правильною командою
5. **Оновлено всі інструкції** для деплою

### 🎯 Рекомендовані Start Commands:

#### 🌐 Тільки веб-дашборд (РЕКОМЕНДОВАНО)
```bash
gunicorn --bind 0.0.0.0:$PORT wsgi:app
```
**Переваги:**
- Стабільний та надійний
- Оптимізований для production
- Мінімальне використання ресурсів
- Ідеально для безкоштовного плану Render

#### 🔄 Повний функціонал (бот + веб)
```bash
python render_runner.py
```
**Включає:**
- Веб-дашборд та API
- 24/7 збір даних
- Sentiment analysis
- Технічний аналіз
- Прогнозування

#### ⚙️ Тільки аналітичний бот
```bash
python main.py
```
**Функції:**
- Збір та аналіз даних
- Без веб-інтерфейсу
- Фонові задачі

### 📁 Ключові файли проекту:
- `wsgi.py` - Production web server entry point
- `render_runner.py` - Full bot + web runner
- `main.py` - Original bot runner
- `Procfile` - Render deployment config
- `requirements.txt` - Python dependencies (з gunicorn)
- `.env.example` - Приклад змінних середовища

### 🌟 Фінальна рекомендація:
Для початку використовуйте **gunicorn** варіант - він найбільш стабільний для безкоштовного хостингу на Render.

---

## 🎯 Що далі?
