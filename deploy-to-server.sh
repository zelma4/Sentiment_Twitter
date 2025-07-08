#!/bin/bash

# Deploy updated files to server and restart
# Usage: ./deploy-to-server.sh [server-ip]

SERVER_IP=${1:-"167.172.105.51"}
SERVER_PATH="/opt/bitcoin-sentiment-bot"
SERVER_USER="root"

echo "🚀 Deploying Bitcoin Sentiment Bot Updates"
echo "=========================================="
echo "Server: $SERVER_IP"
echo "Path: $SERVER_PATH"
echo ""

# Check if we have the files
if [ ! -f "main.py" ] || [ ! -f "config/settings.py" ]; then
    echo "❌ Error: Required files not found. Are you in the project directory?"
    exit 1
fi

echo "📤 Step 1: Uploading updated files..."

# Upload all project files
rsync -avz --progress \
    --exclude='.git' \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='bitcoin_bot.log' \
    --exclude='bitcoin_analysis.db' \
    ./ $SERVER_USER@$SERVER_IP:$SERVER_PATH/

if [ $? -ne 0 ]; then
    echo "❌ Error: File upload failed"
    exit 1
fi

echo ""
echo "🔄 Step 2: Restarting bot on server..."

ssh $SERVER_USER@$SERVER_IP << 'EOF'
cd /opt/bitcoin-sentiment-bot
echo "Current directory: $(pwd)"
echo "Files in directory:"
ls -la | head -10

# Make scripts executable
chmod +x *.sh

# Run update script
if [ -f "update-server.sh" ]; then
    ./update-server.sh
else
    echo "⚠️ update-server.sh not found, running manual update..."
    docker-compose down
    docker-compose build --no-cache
    docker-compose up -d
fi
EOF

echo ""
echo "✅ Deployment complete!"
echo ""
echo "📊 Check status with:"
echo "   ssh $SERVER_USER@$SERVER_IP 'cd $SERVER_PATH && docker-compose logs --tail=20'"
echo ""
echo "🌐 Dashboard: http://$SERVER_IP:8000"
