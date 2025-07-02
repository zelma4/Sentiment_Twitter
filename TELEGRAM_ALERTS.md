# 📱 ЩО НАДСИЛАЄТЬСЯ В TELEGRAM БОТ

## 🔔 Умови для відправки алертів:

### 1. **Значні зміни ціни**
- Прогнозована зміна > 5% на 24 години
- Приклад: якщо Bitcoin може зрости/впасти більше ніж на 5%

### 2. **Екстремальний sentiment**
- Дуже негативний sentiment < -0.5
- Дуже позитивний sentiment > 0.5
- Базується на аналізі Twitter та Reddit постів

### 3. **Сильні технічні сигнали**
- STRONG_BUY (сильний сигнал на купівлю)
- STRONG_SELL (сильний сигнал на продаж)

## 📄 Формат повідомлення:

```
🔮 **Bitcoin Analysis Alert**

💰 **Current Price:** $67,234.56
💭 **Sentiment:** 😊 Positive (0.342)
📊 **Posts Analyzed:** 1,247
📈 **Technical:** STRONG_BUY
🔮 **24h Prediction:** $69,145.23
🎯 **Change:** +2.84%

⏰ **Time:** 2025-07-02 14:30:15 UTC
```

## 🎯 Приклади сценаріїв:

### 🚀 **Bullish Alert**
```
🔮 **Bitcoin Analysis Alert**

💰 **Current Price:** $43,567.89
💭 **Sentiment:** 🚀 Very Positive (0.721)
📊 **Posts Analyzed:** 2,345
📈 **Technical:** STRONG_BUY
🔮 **24h Prediction:** $47,234.12
🎯 **Change:** +8.41%

⏰ **Time:** 2025-07-02 09:15:22 UTC
```

### 📉 **Bearish Alert**
```
🔮 **Bitcoin Analysis Alert**

💰 **Current Price:** $38,123.45
💭 **Sentiment:** 😰 Very Negative (-0.614)
📊 **Posts Analyzed:** 1,893
📈 **Technical:** STRONG_SELL
🔮 **24h Prediction:** $35,789.23
🎯 **Change:** -6.12%

⏰ **Time:** 2025-07-02 16:45:33 UTC
```

### ⚠️ **Mixed Signals Alert**
```
🔮 **Bitcoin Analysis Alert**

💰 **Current Price:** $45,678.90
💭 **Sentiment:** 😐 Neutral (0.034)
📊 **Posts Analyzed:** 987
📈 **Technical:** BUY
🔮 **24h Prediction:** $43,123.45
🎯 **Change:** -5.59%

⏰ **Time:** 2025-07-02 12:20:11 UTC
```

## ⚙️ Налаштування частоти:

- **Не частіше 1 разу на годину** - щоб не спамити
- **Тільки при значних змінах** - фільтр по умовах вище
- **24/7 моніторинг** - працює постійно
- **Автоматично** - без втручання користувача

## 🔧 Як налаштувати:

1. **Створіть Telegram бота:**
   - Напишіть @BotFather
   - Команда `/newbot`
   - Отримайте `TELEGRAM_BOT_TOKEN`

2. **Знайдіть свій Chat ID:**
   - Напишіть боту будь-що
   - Відкрийте: `https://api.telegram.org/bot<YOUR_BOT_TOKEN>/getUpdates`
   - Знайдіть ваш `chat_id`

3. **Додайте в .env:**
   ```
   TELEGRAM_BOT_TOKEN=ваш_токен_бота
   TELEGRAM_CHAT_ID=ваш_chat_id
   ```

## 🎯 Переваги:

- **Миттєві сповіщення** про важливі зміни
- **Структуровані дані** - легко читати
- **Фільтрація** - тільки важливі алерти
- **24/7** - працює постійно
- **Markdown форматування** - красиво виглядає
