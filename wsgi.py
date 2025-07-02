#!/usr/bin/env python3
"""
Production web server entry point for Render deployment.
This file is optimized for web service deployment on Render.
"""

import os
import logging
from web.app import create_app
from database.models import create_database

# Setup logging for production
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Initialize database on startup
try:
    create_database()
    logging.info("Database initialized successfully")
except Exception as e:
    logging.error(f"Database initialization failed: {e}")

# Create Flask app
app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)
