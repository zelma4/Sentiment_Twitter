#!/bin/bash
# Quick setup script for Bitcoin Analysis Bot

echo "🚀 Bitcoin Analysis Bot - Auto Setup"
echo "====================================="

# Check Python version
python3 --version || { echo "❌ Python3 not found. Please install Python 3.8+"; exit 1; }

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Install dependencies
echo "📥 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Create .env from template
if [ ! -f .env ]; then
    echo "⚙️ Creating .env file..."
    cp .env.example .env
    echo "✅ .env file created from template"
else
    echo "✅ .env file already exists"
fi

# Initialize database
echo "🗄️ Initializing database..."
python -c "from database.models import create_database; create_database(); print('Database initialized successfully')"

# Run tests
echo "🧪 Running tests..."
python test_bot.py

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your API keys"
echo "2. Run: source venv/bin/activate"
echo "3. Run: python main.py"
echo "4. Open: http://localhost:8000"
echo ""
echo "For deployment on Render, see DEPLOYMENT.md"
