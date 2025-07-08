#!/bin/bash

# Quick Container Restart Script
# Use this when you've already uploaded new code and just need to restart

echo "🔄 Quick Container Restart"
echo "=========================="

echo "📋 Current status:"
docker ps --format "table {{.Names}}\t{{.Status}}" | grep bitcoin || echo "No bitcoin containers running"

echo ""
echo "🛑 Stopping container..."
docker-compose down

echo ""
echo "🚀 Starting container..."
docker-compose up -d

echo ""
echo "⏳ Waiting for startup..."
sleep 10

echo ""
echo "📊 New status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep bitcoin

echo ""
echo "📋 Recent logs:"
docker-compose logs --tail=10 bitcoin_sentiment_bot

echo ""
echo "✅ Container restarted!"
