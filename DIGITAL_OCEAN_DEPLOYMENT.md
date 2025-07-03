# Розгортання Bitcoin Sentiment Bot на Digital Ocean

## Підготовка

### 1. Створення Droplet на Digital Ocean

1. Зайдіть на [Digital Ocean](https://www.digitalocean.com)
2. Створіть новий Droplet:
   - **Образ**: Ubuntu 22.04 LTS
   - **Plan**: Basic
   - **CPU**: 2 vCPUs, 2GB RAM (мінімум для ML моделей)
   - **Регіон**: найближчий до вас
   - **Authentication**: SSH keys (рекомендовано)

### 2. Підключення до сервера

```bash
ssh root@YOUR_DROPLET_IP
```

## Автоматичне розгортання

### Варіант 1: Використання скрипта

1. Завантажте файли проекту на сервер:
```bash
# На вашому локальному комп'ютері
scp -r /Users/zelma4/Documents/Sentiment_Twitter root@YOUR_DROPLET_IP:/opt/bitcoin-sentiment-bot
```

2. Запустіть скрипт розгортання:
```bash
# На сервері
cd /opt/bitcoin-sentiment-bot
chmod +x deploy.sh
./deploy.sh
```

### Варіант 2: Ручне розгортання

1. **Встановіть Docker та Docker Compose**:
```bash
# Оновіть систему
apt update && apt upgrade -y

# Встановіть Docker
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt update
apt install -y docker-ce docker-ce-cli containerd.io

# Встановіть Docker Compose
curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Запустіть Docker
systemctl start docker
systemctl enable docker
```

2. **Завантажте проект**:
```bash
mkdir -p /opt/bitcoin-sentiment-bot
cd /opt/bitcoin-sentiment-bot
# Завантажте файли проекту
```

3. **Налаштуйте змінні середовища**:
```bash
cp .env.example .env
nano .env
```

Заповніть ваші API ключі:
```env
# Telegram Bot
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Twitter API (опціонально)
TWITTER_BEARER_TOKEN=your_twitter_token

# Reddit API (опціонально)
REDDIT_CLIENT_ID=your_reddit_client_id
REDDIT_CLIENT_SECRET=your_reddit_secret

# Other APIs
FEAR_GREED_API_KEY=your_fear_greed_key
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
```

4. **Запустіть бота**:
```bash
docker-compose up -d
```

## Управління ботом

### Основні команди

```bash
# Запуск бота
docker-compose up -d

# Зупинка бота
docker-compose down

# Перегляд логів
docker-compose logs -f

# Перегляд статусу
docker-compose ps

# Перезапуск бота
docker-compose restart

# Оновлення бота
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

### Моніторинг

```bash
# Перегляд логів реального часу
docker-compose logs -f bitcoin-sentiment-bot

# Перегляд використання ресурсів
docker stats

# Перевірка статусу контейнера
docker ps
```

## Налаштування файрволу

```bash
# Базові правила безпеки
ufw enable
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
ufw allow 22

# Якщо хочете відкрити веб-інтерфейс (опціонально)
# ufw allow 8000
```

## Автоматичні оновлення

Створіть cron job для автоматичного перезапуску:

```bash
crontab -e

# Додайте рядок для перезапуску щодня о 3:00
0 3 * * * cd /opt/bitcoin-sentiment-bot && docker-compose restart
```

## Резервне копіювання

```bash
# Створення резервної копії даних
tar -czf bitcoin-bot-backup-$(date +%Y%m%d).tar.gz /opt/bitcoin-sentiment-bot/data

# Налаштування автоматичного бекапу
echo "0 2 * * * tar -czf /root/backups/bitcoin-bot-backup-\$(date +\%Y\%m\%d).tar.gz /opt/bitcoin-sentiment-bot/data" | crontab -
```

## Усунення несправностей

### Перевірка логів
```bash
# Логи контейнера
docker-compose logs bitcoin-sentiment-bot

# Логи системи
journalctl -u docker.service
```

### Перевірка ресурсів
```bash
# Використання пам'яті та CPU
htop

# Використання диска
df -h
```

### Перезапуск у разі проблем
```bash
docker-compose down
docker system prune -f
docker-compose up -d
```

## Рекомендації по безпеці

1. **Оновлюйте систему регулярно**:
```bash
apt update && apt upgrade -y
```

2. **Використовуйте SSH ключі** замість паролів

3. **Налаштуйте файрвол** (ufw)

4. **Створюйте регулярні бекапи**

5. **Моніторьте логи** на предмет підозрілої активності

## Очікувані витрати

- **Droplet**: $12-24/місяць (2GB RAM)
- **Backup**: $1-2/місяць
- **Load Balancer** (опціонально): $12/місяць

Загалом: **$13-26/місяць** для базового розгортання.
