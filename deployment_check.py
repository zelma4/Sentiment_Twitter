#!/usr/bin/env python3
"""
Deployment Readiness Check for Bitcoin Analysis Bot
Verifies all components are ready for production deployment.
"""

import os
import sys
from pathlib import Path

def check_files():
    """Check if all required files exist"""
    print("📁 Checking required files...")
    
    required_files = [
        'main.py',
        'requirements.txt',
        'Procfile',
        'wsgi.py',
        'build.sh',
        'runtime.txt',
        '.env',
        'config/settings.py',
        'database/models.py',
        'web/app.py',
        'utils/helpers.py'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
        else:
            print(f"✅ {file}")
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    return True

def check_procfile():
    """Check Procfile configuration"""
    print("\n🚀 Checking Procfile...")
    
    try:
        with open('Procfile', 'r') as f:
            content = f.read().strip()
        
        if 'gunicorn' in content and 'wsgi:app' in content:
            print(f"✅ Procfile: {content}")
            return True
        else:
            print(f"❌ Procfile content issue: {content}")
            return False
            
    except Exception as e:
        print(f"❌ Procfile error: {e}")
        return False

def check_requirements():
    """Check requirements.txt"""
    print("\n📦 Checking requirements...")
    
    try:
        with open('requirements.txt', 'r') as f:
            content = f.read()
        
        required_packages = ['gunicorn', 'fastapi', 'requests', 'pandas', 'ta']
        missing_packages = []
        
        for package in required_packages:
            if package not in content:
                missing_packages.append(package)
            else:
                print(f"✅ {package}")
        
        if missing_packages:
            print(f"❌ Missing packages: {missing_packages}")
            return False
        
        # Check for problematic packages
        if 'TA-Lib' in content and not content.count('# TA-Lib'):
            print("⚠️ Warning: TA-Lib should be commented out for Render")
        
        return True
        
    except Exception as e:
        print(f"❌ Requirements error: {e}")
        return False

def check_environment():
    """Check environment configuration"""
    print("\n🔧 Checking environment...")
    
    try:
        from config.settings import settings
        
        # Critical settings
        checks = [
            ('UPDATE_INTERVAL_MINUTES', settings.UPDATE_INTERVAL_MINUTES in [10, 30]),  # Allow both values
            ('Database URL', settings.DATABASE_URL is not None),
            ('Port', settings.PORT == 8000),
            ('Host', settings.HOST == '0.0.0.0')
        ]
        
        all_good = True
        for name, check in checks:
            if check:
                print(f"✅ {name}")
            else:
                print(f"❌ {name}")
                all_good = False
        
        # Optional but recommended
        optional_checks = [
            ('Telegram Bot Token', bool(settings.TELEGRAM_BOT_TOKEN)),
            ('Twitter Bearer Token', bool(settings.TWITTER_BEARER_TOKEN)),
            ('Reddit Client ID', bool(settings.REDDIT_CLIENT_ID))
        ]
        
        print("\n📱 Optional configurations:")
        for name, check in optional_checks:
            status = "✅" if check else "⚠️"
            print(f"{status} {name}")
        
        return all_good
        
    except Exception as e:
        print(f"❌ Environment error: {e}")
        return False

def check_build_script():
    """Check build.sh script"""
    print("\n🔨 Checking build script...")
    
    try:
        with open('build.sh', 'r') as f:
            content = f.read()
        
        if 'pip install -r requirements.txt' in content:
            print("✅ Build script looks good")
            return True
        else:
            print("❌ Build script missing pip install")
            return False
            
    except Exception as e:
        print(f"❌ Build script error: {e}")
        return False

def check_gitignore():
    """Check .gitignore configuration"""
    print("\n🚫 Checking .gitignore...")
    
    try:
        with open('.gitignore', 'r') as f:
            content = f.read()
        
        important_ignores = ['.env', '__pycache__', '*.log', '.venv']
        missing_ignores = []
        
        for ignore in important_ignores:
            if ignore not in content:
                missing_ignores.append(ignore)
            else:
                print(f"✅ {ignore}")
        
        # Check for Python cache patterns (*.pyc OR *.py[cod])
        if '*.pyc' in content or '*.py[cod]' in content:
            print("✅ Python cache files")
        else:
            missing_ignores.append('*.pyc or *.py[cod]')
        
        if missing_ignores:
            print(f"⚠️ Missing from .gitignore: {missing_ignores}")
        
        return len(missing_ignores) == 0
        
    except Exception as e:
        print(f"❌ .gitignore error: {e}")
        return False

def deployment_summary():
    """Print deployment summary"""
    print("\n" + "="*60)
    print("🚀 DEPLOYMENT INSTRUCTIONS")
    print("="*60)
    print("""
1. Commit all changes to Git:
   git add .
   git commit -m "Ready for deployment"
   git push origin main

2. Deploy to Render:
   - Connect your GitHub repository
   - Use "Web Service" type
   - Build Command: ./build.sh
   - Start Command: gunicorn wsgi:app --host 0.0.0.0 --port $PORT
   - Environment: Add your .env variables

3. Environment Variables needed on Render:
   - TELEGRAM_BOT_TOKEN (for alerts)
   - TELEGRAM_CHAT_ID (for alerts)
   - TWITTER_BEARER_TOKEN (for data collection)
   - REDDIT_CLIENT_ID (for data collection)
   - REDDIT_CLIENT_SECRET (for data collection)
   - UPDATE_INTERVAL_MINUTES=10
   - DATABASE_URL (will auto-generate SQLite)

4. Monitor deployment:
   - Check logs for startup messages
   - Test web dashboard at your-app.onrender.com
   - Verify Telegram alerts are working

5. Bot Features:
   - Collects data every 10 minutes
   - Critical alerts sent immediately  
   - Regular updates every 10 minutes (with data)
   - Hourly summaries every 2 hours
   - Web dashboard at / (simple) and /full (detailed)
""")

def main():
    """Run all deployment checks"""
    print("🔍 Bitcoin Analysis Bot - Deployment Readiness Check")
    print("="*60)
    
    checks = [
        check_files(),
        check_procfile(),
        check_requirements(),
        check_environment(),
        check_build_script(),
        check_gitignore()
    ]
    
    all_passed = all(checks)
    
    print("\n" + "="*60)
    if all_passed:
        print("✅ DEPLOYMENT READY! All checks passed.")
        deployment_summary()
    else:
        print("❌ DEPLOYMENT NOT READY! Fix issues above.")
        
    print("\n📋 Quick Test: Run 'python test_imports.py' to verify components")
    print("🏃 Start Bot: Run 'python main.py' to test locally")
    
    return all_passed

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
