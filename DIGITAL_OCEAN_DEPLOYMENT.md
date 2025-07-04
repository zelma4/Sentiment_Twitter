# Bitcoin Sentiment Bot - Production Deployment

## Quick Start

### 1. Create Digital Ocean Droplet
- **OS**: Ubuntu 22.04 LTS
- **Size**: 2GB RAM, 1 vCPU ($12/month)
- **Region**: Choose nearest
- **Authentication**: SSH keys

### 2. Deploy to Server

```bash
# On your local machine - copy files to server
scp -r /path/to/project root@YOUR_SERVER_IP:/opt/bitcoin-sentiment-bot

# SSH to server
ssh root@YOUR_SERVER_IP

# Run deployment script
cd /opt/bitcoin-sentiment-bot
chmod +x deploy.sh
./deploy.sh
```

### 3. Configure Environment

```bash
# Create .env file
nano .env
```

Add your API keys:
```env
TELEGRAM_BOT_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_telegram_chat_id
```

### 4. Start the Bot

```bash
# Start the bot
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

## Management Commands

```bash
# Start
docker-compose up -d

# Stop
docker-compose down

# Restart
docker-compose restart

# View logs
docker-compose logs -f

# Update (after code changes)
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Monitoring

```bash
# Real-time logs
docker-compose logs -f

# Resource usage
docker stats

# Container status
docker ps
```

## Troubleshooting

### Build Issues
```bash
# Clean rebuild
docker-compose down
docker system prune -f
docker-compose build --no-cache
docker-compose up -d
```

### Memory Issues
```bash
# Check memory usage
free -h

# Restart if needed
docker-compose restart
```

### Network Issues
```bash
# Check connectivity
curl -I https://api.telegram.org
```

## Security

```bash
# Setup firewall
ufw enable
ufw default deny incoming
ufw default allow outgoing
ufw allow ssh
```

## Backup

```bash
# Create backup
tar -czf backup-$(date +%Y%m%d).tar.gz /opt/bitcoin-sentiment-bot/data
```

## Cost Estimation

- **Droplet**: $12/month (2GB RAM)
- **Storage**: $1/month
- **Total**: ~$13/month
