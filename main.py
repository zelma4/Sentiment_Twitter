#!/usr/bin/env python3
"""
Bitcoin Analysis Bot
Multi-functional bot for Bitcoin sentiment analysis, technical analysis, and price prediction.
"""

import logging
import schedule
import time
import asyncio
from datetime import datetime
from config.settings import settings
from database.models import create_database
from data_collectors.twitter_collector import TwitterCollector
from data_collectors.reddit_collector import RedditCollector
from data_collectors.price_collector import PriceCollector
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.technical_analyzer import TechnicalAnalyzer
from analysis.predictor import BitcoinPredictor
import threading

class BitcoinAnalysisBot:
    def __init__(self):
        self.setup_logging()
        self.initialize_components()
        self.running = False
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('bitcoin_bot.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_components(self):
        """Initialize all bot components"""
        self.logger.info("Initializing Bitcoin Analysis Bot...")
        
        # Initialize database
        try:
            create_database()
            self.logger.info("Database initialized successfully")
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            
        # Initialize collectors
        self.twitter_collector = TwitterCollector()
        self.reddit_collector = RedditCollector()
        self.price_collector = PriceCollector()
        
        # Initialize analyzers
        self.sentiment_analyzer = SentimentAnalyzer()
        self.technical_analyzer = TechnicalAnalyzer()
        self.predictor = BitcoinPredictor()
        
        self.logger.info("All components initialized")
    
    def collect_data(self):
        """Collect data from all sources"""
        self.logger.info("=== Starting Data Collection Cycle ===")
        
        try:
            # Collect price data
            self.logger.info("Collecting price data...")
            price_stats = self.price_collector.collect_and_save()
            
            # Collect Twitter data
            self.logger.info("Collecting Twitter data...")
            twitter_stats = self.twitter_collector.collect_and_save(max_results=100)
            
            # Collect Reddit data
            self.logger.info("Collecting Reddit data...")
            reddit_stats = self.reddit_collector.collect_and_save(posts_limit=50, comments_limit=30)
            
            # Log collection summary
            total_social_posts = 0
            if twitter_stats:
                total_social_posts += twitter_stats.get('saved_tweets', 0)
            if reddit_stats:
                total_social_posts += reddit_stats.get('saved_entries', 0)
                
            self.logger.info(f"Data collection complete - Social posts: {total_social_posts}")
            
            return {
                'price': price_stats,
                'twitter': twitter_stats,
                'reddit': reddit_stats,
                'total_social_posts': total_social_posts
            }
            
        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
            return None
    
    def perform_analysis(self):
        """Perform sentiment and technical analysis"""
        self.logger.info("=== Starting Analysis Cycle ===")
        
        try:
            # Perform sentiment analysis
            self.logger.info("Performing sentiment analysis...")
            sentiment_summary = self.sentiment_analyzer.generate_sentiment_summary(hours=24)
            
            # Perform technical analysis
            self.logger.info("Performing technical analysis...")
            technical_summary = self.technical_analyzer.perform_full_analysis(days=30)
            
            # Generate predictions
            self.logger.info("Generating price predictions...")
            prediction_report = self.predictor.generate_prediction_report()
            
            # Create comprehensive analysis report
            analysis_report = {
                'timestamp': datetime.utcnow(),
                'sentiment': sentiment_summary,
                'technical': technical_summary,
                'predictions': prediction_report
            }
            
            self.logger.info("Analysis cycle complete")
            
            # Log key insights
            self.log_key_insights(analysis_report)
            
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {e}")
            return None
    
    def log_key_insights(self, analysis_report):
        """Log key insights from analysis"""
        try:
            # Sentiment insights
            if analysis_report.get('sentiment'):
                sentiment = analysis_report['sentiment']['overall']
                self.logger.info(f"SENTIMENT: {sentiment['overall_score']:.3f} "
                               f"({sentiment['total_posts']} posts, "
                               f"{sentiment['confidence']:.2f} confidence)")
            
            # Technical insights
            if analysis_report.get('technical'):
                technical = analysis_report['technical']
                self.logger.info(f"TECHNICAL: {technical['recommendation']} "
                               f"(Strength: {technical['signals']['strength']:.2f})")
            
            # Prediction insights
            if analysis_report.get('predictions'):
                predictions = analysis_report['predictions']['predictions']
                if '24h' in predictions:
                    pred_24h = predictions['24h']
                    self.logger.info(f"PREDICTION (24h): ${pred_24h['predicted_price']:.2f} "
                                   f"({pred_24h['price_change_pct']:+.2f}%, "
                                   f"{pred_24h['confidence']:.2f} confidence)")
                    
        except Exception as e:
            self.logger.error(f"Error logging insights: {e}")
    
    def retrain_model_weekly(self):
        """Retrain the ML model weekly"""
        self.logger.info("=== Weekly Model Retraining ===")
        
        try:
            success = self.predictor.retrain_model()
            if success:
                self.logger.info("Model retrained successfully")
            else:
                self.logger.error("Model retraining failed")
                
        except Exception as e:
            self.logger.error(f"Error in model retraining: {e}")
    
    def run_full_cycle(self):
        """Run a complete analysis cycle"""
        try:
            self.logger.info("🚀 Starting Bitcoin Analysis Cycle")
            
            # Step 1: Collect data
            collection_stats = self.collect_data()
            
            # Step 2: Perform analysis (only if we have data)
            if collection_stats and collection_stats.get('total_social_posts', 0) > 0:
                analysis_report = self.perform_analysis()
                
                if analysis_report:
                    self.logger.info("✅ Analysis cycle completed successfully")
                else:
                    self.logger.warning("⚠️ Analysis cycle completed with errors")
            else:
                self.logger.warning("⚠️ Skipping analysis - insufficient data collected")
                
        except Exception as e:
            self.logger.error(f"❌ Error in full cycle: {e}")
    
    def setup_scheduler(self):
        """Setup scheduled tasks"""
        self.logger.info("Setting up scheduler...")
        
        # Data collection and analysis every 30 minutes
        schedule.every(settings.UPDATE_INTERVAL_MINUTES).minutes.do(self.run_full_cycle)
        
        # Model retraining every week
        schedule.every().sunday.at("02:00").do(self.retrain_model_weekly)
        
        # Initial data collection with historical data
        schedule.every().day.at("01:00").do(
            lambda: self.price_collector.collect_and_save(include_historical=True, historical_days=7)
        )
        
        self.logger.info("Scheduler configured")
    
    def start_web_server(self):
        """Start the web dashboard in a separate thread"""
        try:
            from web.app import create_app
            app = create_app()
            
            def run_server():
                app.run(host=settings.HOST, port=settings.PORT, debug=False)
            
            web_thread = threading.Thread(target=run_server, daemon=True)
            web_thread.start()
            
            self.logger.info(f"Web server started on http://{settings.HOST}:{settings.PORT}")
            
        except Exception as e:
            self.logger.error(f"Failed to start web server: {e}")
    
    def run(self):
        """Main run method"""
        self.logger.info("🚀 Bitcoin Analysis Bot Starting Up")
        
        # Initial setup
        self.setup_scheduler()
        
        # Start web server
        self.start_web_server()
        
        # Run initial cycle
        self.logger.info("Running initial analysis cycle...")
        self.run_full_cycle()
        
        # Start scheduled tasks
        self.running = True
        self.logger.info("🔄 Bot is now running 24/7")
        self.logger.info(f"⏰ Next analysis in {settings.UPDATE_INTERVAL_MINUTES} minutes")
        
        try:
            while self.running:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
                
        except KeyboardInterrupt:
            self.logger.info("⏹️ Bot stopped by user")
            self.running = False
        except Exception as e:
            self.logger.error(f"❌ Bot stopped due to error: {e}")
            self.running = False
    
    def stop(self):
        """Stop the bot"""
        self.logger.info("Stopping Bitcoin Analysis Bot...")
        self.running = False

def main():
    """Main entry point"""
    print("🔮 Bitcoin Analysis Bot v1.0")
    print("=" * 50)
    
    # Initialize and run bot
    bot = BitcoinAnalysisBot()
    
    try:
        bot.run()
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
        bot.stop()
    except Exception as e:
        print(f"❌ Bot crashed: {e}")
        logging.error(f"Bot crashed: {e}")

if __name__ == "__main__":
    main()
