#!/usr/bin/env python3
"""
Web Dashboard Runner for Bitcoin Analysis Bot
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from web.app import create_app

if __name__ == "__main__":
    print("🌐 Starting Bitcoin Analysis Dashboard")
    print("=" * 50)
    print("📊 Dashboard: http://localhost:8000")
    print("📈 Enhanced: http://localhost:8000/enhanced")
    print("🔄 API: http://localhost:8000/api/current-analysis")
    print("=" * 50)
    
    app = create_app()
    app.run(debug=False, host='0.0.0.0', port=8000)
