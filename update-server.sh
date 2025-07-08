#!/bin/bash

# Bitcoin Sentiment Bot - Server Update Script
# This script properly updates and restarts the Docker container

echo "🔄 Bitcoin Sentiment Bot - Server Update"
echo "========================================"

# Check if we're in the correct directory
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ Error: docker-compose.yml not found. Are you in the right directory?"
    exit 1
fi

echo "📋 Current container status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep bitcoin || echo "No bitcoin containers running"

echo ""
echo "🛑 Step 1: Stopping existing containers..."
docker-compose down --remove-orphans

echo ""
echo "🧹 Step 2: Cleaning up old images and cache..."
# Remove old images
docker image prune -f
# Remove old build cache
docker builder prune -f

echo ""
echo "🔨 Step 3: Building new image (no cache)..."
docker-compose build --no-cache

echo ""
echo "🚀 Step 4: Starting new container..."
docker-compose up -d

echo ""
echo "⏳ Step 5: Waiting for container to start..."
sleep 10

echo ""
echo "📊 Step 6: Checking container status..."
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep bitcoin

echo ""
echo "📋 Step 7: Checking logs for errors..."
docker-compose logs --tail=20 bitcoin_sentiment_bot

echo ""
echo "🌐 Step 8: Testing dashboard availability..."
sleep 5
curl -s http://localhost:8000 > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Dashboard is accessible at http://localhost:8000"
else
    echo "⚠️  Dashboard might not be ready yet (give it a minute)"
fi

echo ""
echo "✅ Update complete!"
echo ""
echo "📱 Useful commands:"
echo "   View logs:     docker-compose logs -f bitcoin_sentiment_bot"
echo "   Stop bot:      docker-compose down"
echo "   Restart:       docker-compose restart"
echo "   Check status:  docker ps"
echo ""
echo "🌐 Dashboard: http://YOUR_SERVER_IP:8000"
