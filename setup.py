#!/usr/bin/env python3
"""
Local setup script for Bitcoin Analysis Bot
"""

import os
import subprocess
import sys
from pathlib import Path

def run_command(command, description):
    """Run a shell command and handle errors"""
    print(f"📦 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {sys.version.split()[0]} detected")
    return True

def setup_virtual_environment():
    """Create and activate virtual environment"""
    if not os.path.exists("venv"):
        print("🔧 Creating virtual environment...")
        if not run_command("python -m venv venv", "Virtual environment creation"):
            return False
    
    # Check if we're in virtual environment
    if sys.prefix == sys.base_prefix:
        print("⚠️  Please activate the virtual environment:")
        if os.name == 'nt':  # Windows
            print("   venv\\Scripts\\activate")
        else:  # Unix/Linux/Mac
            print("   source venv/bin/activate")
        return False
    
    print("✅ Virtual environment is active")
    return True

def install_dependencies():
    """Install Python dependencies"""
    return run_command("pip install -r requirements.txt", "Installing dependencies")

def setup_environment_file():
    """Create .env file from template"""
    if not os.path.exists(".env"):
        if os.path.exists(".env.example"):
            run_command("cp .env.example .env", "Creating .env file")
            print("⚠️  Please edit .env file with your API keys")
        else:
            print("❌ .env.example not found")
            return False
    else:
        print("✅ .env file already exists")
    return True

def initialize_database():
    """Initialize the database"""
    try:
        from database.models import create_database
        create_database()
        print("✅ Database initialized successfully")
        return True
    except Exception as e:
        print(f"❌ Database initialization failed: {e}")
        return False

def test_components():
    """Test basic functionality"""
    print("🧪 Testing components...")
    
    # Test database connection
    try:
        from database.models import get_session
        session = get_session()
        session.close()
        print("✅ Database connection test passed")
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False
    
    # Test price collector (without API keys)
    try:
        from data_collectors.price_collector import PriceCollector
        collector = PriceCollector()
        print("✅ Price collector initialized")
    except Exception as e:
        print(f"⚠️  Price collector test failed: {e}")
    
    return True

def main():
    """Main setup function"""
    print("🚀 Bitcoin Analysis Bot Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        return False
    
    # Setup virtual environment
    if not setup_virtual_environment():
        return False
    
    # Install dependencies
    if not install_dependencies():
        return False
    
    # Setup environment file
    if not setup_environment_file():
        return False
    
    # Initialize database
    if not initialize_database():
        return False
    
    # Test components
    if not test_components():
        print("⚠️  Some components failed tests, but setup can continue")
    
    print("\n🎉 Setup completed successfully!")
    print("\nNext steps:")
    print("1. Edit .env file with your API keys")
    print("2. Run: python main.py")
    print("3. Open http://localhost:8000 in your browser")
    print("\nFor deployment on Render:")
    print("1. Push code to GitHub")
    print("2. Connect Render to your repository")
    print("3. Set environment variables in Render dashboard")
    
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⏹️  Setup interrupted by user")
    except Exception as e:
        print(f"\n❌ Setup failed: {e}")
        sys.exit(1)
