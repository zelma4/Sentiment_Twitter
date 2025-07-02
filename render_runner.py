#!/usr/bin/env python3
"""
Full bot runner for production deployment.
Runs both the analysis bot and web server in separate threads.
"""

import os
import threading
import logging
from main import BitcoinAnalysisBot
from web.app import create_app
from database.models import create_database


def run_web_server():
    """Run Flask web server in a separate thread"""
    app = create_app()
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)


def run_analysis_bot():
    """Run analysis bot in a separate thread"""
    bot = BitcoinAnalysisBot()
    bot.run()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize database
    try:
        create_database()
        logging.info("Database initialized successfully")
    except Exception as e:
        logging.error(f"Database initialization failed: {e}")
    
    # Start web server in a separate thread
    web_thread = threading.Thread(target=run_web_server, daemon=True)
    web_thread.start()
    
    # Start analysis bot in main thread
    try:
        run_analysis_bot()
    except KeyboardInterrupt:
        logging.info("Application stopped by user")
    except Exception as e:
        logging.error(f"Application crashed: {e}")
