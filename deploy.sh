#!/bin/bash

# Bitcoin Sentiment Bot Deployment Script for Digital Ocean
# Make sure to run this script as root or with sudo

echo "🚀 Starting Bitcoin Sentiment Bot deployment on Digital Ocean..."

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Docker
echo "🐳 Installing Docker..."
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor -o /usr/share/keyrings/docker-archive-keyring.gpg
echo "deb [arch=amd64 signed-by=/usr/share/keyrings/docker-archive-keyring.gpg] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | tee /etc/apt/sources.list.d/docker.list > /dev/null
apt update
apt install -y docker-ce docker-ce-cli containerd.io

# Install Docker Compose
echo "🔧 Installing Docker Compose..."
curl -L "https://github.com/docker/compose/releases/download/v2.21.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose

# Start Docker service
echo "▶️ Starting Docker service..."
systemctl start docker
systemctl enable docker

# Create app directory
echo "📁 Creating application directory..."
mkdir -p /opt/bitcoin-sentiment-bot
cd /opt/bitcoin-sentiment-bot

# Clone or copy your project here
# git clone <your-repo-url> .

echo "✅ Digital Ocean environment setup complete!"
echo "📋 Next steps:"
echo "1. Copy your project files to /opt/bitcoin-sentiment-bot/"
echo "2. Create .env file with your API keys"
echo "3. Run: docker-compose up -d"
echo "4. Check logs: docker-compose logs -f"
