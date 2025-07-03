#!/usr/bin/env python3
"""
Bitcoin Analysis Bot
Multi-functional bot for Bitcoin sentiment analysis, technical analysis, and price prediction.
Enhanced with CryptoBERT, LightGBM, and additional data sources.
"""

import logging
import schedule
import time
import asyncio
from datetime import datetime, timedelta
from config.settings import settings
from database.models import create_database
from data_collectors.twitter_collector import TwitterCollector
from data_collectors.reddit_collector import RedditCollector
from data_collectors.price_collector import PriceCollector
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.technical_analyzer import TechnicalAnalyzer
from analysis.predictor import BitcoinPredictor
from utils.helpers import (
    send_telegram_message, 
    create_enhanced_alert_message
)
import threading

# Enhanced components
try:
    from data_collectors.enhanced_collector import EnhancedDataCollector
    from analysis.lightgbm_predictor import LightGBMPredictor
    ENHANCED_FEATURES_AVAILABLE = True
except ImportError:
    ENHANCED_FEATURES_AVAILABLE = False

class BitcoinAnalysisBot:
    def __init__(self):
        self.setup_logging()
        self.initialize_components()
        self.running = False
        self.last_alert_time = None  # Track last alert time
        self.startup_message_sent = False  # Track startup message
        self.last_enhanced_metrics = {}  # Store last enhanced metrics
        
        # Enhanced features
        if ENHANCED_FEATURES_AVAILABLE:
            self.enhanced_collector = EnhancedDataCollector()
            self.lightgbm_predictor = LightGBMPredictor()
            self.logger.info("🚀 Enhanced features enabled (CryptoBERT + LightGBM)")
        else:
            self.enhanced_collector = None
            self.lightgbm_predictor = None
            self.logger.info("📊 Running with standard features only")
        
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
        """Collect data from all sources including enhanced metrics"""
        self.logger.info("=== Starting Enhanced Data Collection Cycle ===")
        
        try:
            # Collect price data
            self.logger.info("Collecting price data...")
            price_stats = self.price_collector.collect_and_save()
            
            # Collect Reddit data first (always works)
            self.logger.info("Collecting Reddit data...")
            reddit_stats = self.reddit_collector.collect_and_save(
                posts_limit=20, 
                comments_limit=10
            )
            
            # Collect Twitter data (skip if rate limited)
            self.logger.info("Collecting Twitter data...")
            try:
                twitter_stats = self.twitter_collector.collect_and_save(
                    max_results=25
                )
                self.logger.info("Twitter collection successful")
            except Exception as e:
                if "Rate limit" in str(e) or "Sleeping for" in str(e):
                    self.logger.warning(
                        "Twitter rate limited - skipping for this cycle"
                    )
                    twitter_stats = None
                else:
                    self.logger.warning(f"Twitter collection failed: {e}")
                    twitter_stats = None
            
            # Collect enhanced metrics if available
            enhanced_metrics = {}
            if self.enhanced_collector:
                self.logger.info("Collecting enhanced market metrics...")
                try:
                    # Get recent price data for correlations
                    recent_prices = self.price_collector.get_recent_prices(days=5)
                    enhanced_metrics = self.enhanced_collector.collect_all_metrics(
                        btc_prices=recent_prices
                    )
                    self.last_enhanced_metrics = enhanced_metrics  # Store for analysis
                    self.logger.info(
                        f"✅ Enhanced metrics collected: {len(enhanced_metrics)} indicators"
                    )
                except Exception as e:
                    self.logger.warning(f"Enhanced metrics collection failed: {e}")
                    enhanced_metrics = {}
            
            # Log collection summary
            total_social_posts = 0
            if twitter_stats:
                total_social_posts += twitter_stats.get('saved_tweets', 0)
            if reddit_stats:
                total_social_posts += reddit_stats.get('saved_entries', 0)
                
            self.logger.info(
                f"Data collection complete - Social posts: {total_social_posts}, "
                f"Enhanced metrics: {len(enhanced_metrics)}"
            )
            
            return {
                'price': price_stats,
                'twitter': twitter_stats,
                'reddit': reddit_stats,
                'enhanced_metrics': enhanced_metrics,
                'total_social_posts': total_social_posts
            }
            
        except Exception as e:
            self.logger.error(f"Error in data collection: {e}")
            return None
    
    def perform_analysis(self):
        """Perform enhanced sentiment and technical analysis"""
        self.logger.info("=== Starting Enhanced Analysis Cycle ===")
        
        try:
            # Perform sentiment analysis (now using CryptoBERT if available)
            self.logger.info("Performing advanced sentiment analysis...")
            sentiment_summary = self.sentiment_analyzer.generate_sentiment_summary(
                hours=24
            )
            
            # Perform technical analysis
            self.logger.info("Performing technical analysis...")
            technical_summary = self.technical_analyzer.perform_full_analysis(
                days=30
            )
            
            # Generate standard predictions
            self.logger.info("Generating standard price predictions...")
            prediction_report = self.predictor.generate_prediction_report()
            
            # Generate LightGBM predictions if available
            lightgbm_prediction = None
            if self.lightgbm_predictor:
                self.logger.info("Generating LightGBM price direction prediction...")
                try:
                    # Get recent price and sentiment data for LightGBM
                    recent_prices = self.price_collector.get_recent_prices(days=60)
                    recent_sentiment = self.sentiment_analyzer.get_recent_sentiment_data(hours=1440)  # 60 days
                    
                    lightgbm_prediction = self.lightgbm_predictor.predict_next_direction(
                        price_data=recent_prices,
                        sentiment_data=recent_sentiment
                    )
                    
                    if lightgbm_prediction:
                        self.logger.info(
                            f"LightGBM prediction: {lightgbm_prediction['direction_text']} "
                            f"(confidence: {lightgbm_prediction['confidence']:.3f})"
                        )
                    else:
                        self.logger.warning("LightGBM prediction failed")
                        
                except Exception as e:
                    self.logger.error(f"LightGBM prediction failed: {e}")
                    lightgbm_prediction = None
            
            # Create comprehensive analysis report
            analysis_report = {
                'timestamp': datetime.utcnow(),
                'sentiment': sentiment_summary,
                'technical': technical_summary,
                'predictions': prediction_report,
                'lightgbm_prediction': lightgbm_prediction,
                'enhanced_metrics': self.last_enhanced_metrics,
                'price_data': self._get_current_price_data()
            }
            
            self.logger.info("Analysis cycle complete")
            
            # Log key insights
            self.log_key_insights(analysis_report)
            
            # Send Telegram alert if conditions are met
            if self.should_send_alert(analysis_report):
                self.send_telegram_alert(analysis_report)
            
            return analysis_report
            
        except Exception as e:
            self.logger.error(f"Error in analysis: {e}")
            return None
    
    def _get_current_price_data(self):
        """Get current price data for analysis report"""
        try:
            current_price = self.price_collector.get_current_price()
            # Simple implementation - could be enhanced with 24h change calculation
            return {
                'current_price': current_price,
                'price_change_24h': 0  # Placeholder - would need historical comparison
            }
        except Exception as e:
            self.logger.error(f"Error getting current price data: {e}")
            return {
                'current_price': None,
                'price_change_24h': 0
            }

    def _prepare_lightgbm_data(self):
        """Prepare data for LightGBM prediction"""
        try:
            # Get recent price data
            price_data = self.price_collector.get_recent_prices(days=60)
            if price_data is None or price_data.empty:
                return None
            
            # Get recent sentiment data  
            sentiment_data = self.sentiment_analyzer.get_recent_sentiment_data(
                hours=24*60  # 60 days
            )
            
            # Convert sentiment data to daily aggregates
            if sentiment_data:
                import pandas as pd
                
                sentiment_df = pd.DataFrame([{
                    'date': item.timestamp.date(),
                    'sentiment': item.sentiment_score
                } for item in sentiment_data])
                
                # Group by date and take mean
                sentiment_daily = sentiment_df.groupby('date')['sentiment'].mean()
                sentiment_daily = sentiment_daily.to_frame().reset_index()
                sentiment_daily.set_index('date', inplace=True)
            else:
                sentiment_daily = pd.DataFrame()
            
            # Prepare features using LightGBM predictor
            if self.lightgbm_predictor:
                features_df = self.lightgbm_predictor.prepare_features(
                    price_data, 
                    sentiment_daily
                )
                return features_df
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to prepare LightGBM data: {e}")
            return None
    
    def generate_enhanced_alert(self, analysis_report, collection_stats):
        """Generate enhanced alert with CryptoBERT and LightGBM insights"""
        try:
            # Get current price
            current_price = self.price_collector.get_current_price()
            if current_price is None:
                current_price = 0.0
            
            # Base alert info
            alert_parts = [
                "🤖 Enhanced Bitcoin Analysis Alert",
                f"💰 Current Price: ${current_price:,.2f}",
                f"📊 Time: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
            ]
            
            # Handle None analysis_report
            if analysis_report is None:
                analysis_report = {}
            
            # Sentiment analysis (enhanced with CryptoBERT)
            sentiment = analysis_report.get('sentiment', {})
            if sentiment:
                sentiment_score = sentiment.get('overall_score', 0)
                emoji = "🟢" if sentiment_score > 0.1 else "🔴" if sentiment_score < -0.1 else "🟡"
                
                # Check if CryptoBERT was used
                analyzer = getattr(self, 'sentiment_analyzer', None)
                if analyzer and hasattr(analyzer, 'use_advanced') and analyzer.use_advanced:
                    alert_parts.append(f"{emoji} CryptoBERT Sentiment: {sentiment_score:.3f}")
                else:
                    alert_parts.append(f"{emoji} Sentiment: {sentiment_score:.3f}")
            
            # LightGBM prediction
            if 'lightgbm_prediction' in analysis_report:
                lgb_pred = analysis_report['lightgbm_prediction']
                if lgb_pred and isinstance(lgb_pred, dict):
                    direction = lgb_pred.get('direction', 0)
                    direction_emoji = "⬆️" if direction == 1 else "⬇️"
                    direction_text = lgb_pred.get('direction_text', 'Unknown')
                    confidence = lgb_pred.get('confidence', 0)
                    
                    alert_parts.append(
                        f"🧠 LightGBM: {direction_emoji} {direction_text} "
                        f"(Conf: {confidence:.1%})"
                    )
            
            # Enhanced metrics
            if collection_stats is None:
                collection_stats = {}
            
            enhanced_metrics = collection_stats.get('enhanced_metrics', {})
            if enhanced_metrics:
                # Fear & Greed
                if 'fear_greed_value' in enhanced_metrics:
                    fg_value = enhanced_metrics['fear_greed_value']
                    fg_class = enhanced_metrics.get('fear_greed_class', 'Unknown')
                    alert_parts.append(f"😨 Fear & Greed: {fg_value} ({fg_class})")
                
                # StockTwits sentiment
                if 'stocktwits_sentiment' in enhanced_metrics:
                    st_sentiment = enhanced_metrics['stocktwits_sentiment']
                    st_emoji = "🟢" if st_sentiment > 0.1 else "🔴" if st_sentiment < -0.1 else "🟡"
                    alert_parts.append(f"{st_emoji} StockTwits: {st_sentiment:.3f}")
            
            # Technical analysis
            technical = analysis_report.get('technical', {})
            if technical:
                trend = technical.get('trend', 'Unknown')
                trend_emoji = "📈" if trend == "Bullish" else "📉" if trend == "Bearish" else "➡️"
                alert_parts.append(f"{trend_emoji} Technical: {trend}")
            
            # Data collection stats
            social_posts = collection_stats.get('total_social_posts', 0)
            enhanced_count = len(enhanced_metrics) if enhanced_metrics else 0
            alert_parts.append(f"📱 Social: {social_posts} | Enhanced: {enhanced_count}")
            
            return "\n".join(alert_parts)
            
        except Exception as e:
            self.logger.error(f"Error generating enhanced alert: {e}")
            return self.generate_fallback_alert(analysis_report, collection_stats)
    
    def generate_fallback_alert(self, analysis_report, collection_stats):
        """Generate basic alert if enhanced features fail"""
        try:
            current_price = self.price_collector.get_current_price()
            sentiment = analysis_report.get('sentiment', {})
            sentiment_score = sentiment.get('overall_score', 0)
            
            return (
                f"🤖 Bitcoin Analysis Alert\n"
                f"💰 Price: ${current_price:,.2f}\n"
                f"📊 Sentiment: {sentiment_score:.3f}\n"
                f"⏰ {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}"
            )
        except:
            return "🤖 Bitcoin Analysis Alert - Data temporarily unavailable"
    
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
                    
            # LightGBM prediction insights
            if analysis_report.get('lightgbm_prediction'):
                lg_prediction = analysis_report['lightgbm_prediction']
                self.logger.info(f"LIGHTGBM PREDICTION: {lg_prediction['direction_text']} "
                               f"(Confidence: {lg_prediction['confidence']:.2f})")
                    
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
                    # Step 3: Check if we should send alerts
                    self.check_and_send_alerts(analysis_report)
                    
                    self.logger.info("✅ Analysis cycle completed successfully")
                else:
                    self.logger.warning("⚠️ Analysis cycle completed with errors")
            else:
                self.logger.warning("⚠️ Skipping analysis - insufficient data collected")
                
        except Exception as e:
            self.logger.error(f"❌ Error in full cycle: {e}")

    def check_and_send_alerts(self, analysis_report):
        """Check if alerts should be sent and send them"""
        try:
            # Always check for critical alerts first (highest priority)
            if self.is_critical_alert(analysis_report):
                if self.should_send_alert(analysis_report, force_alert=False):
                    self.send_telegram_alert(analysis_report, alert_type="CRITICAL")
                    self.logger.info("🚨 Critical alert sent")
                    return
            
            # Check for regular updates based on data significance
            if self.should_send_regular_update(analysis_report):
                if self.should_send_alert(analysis_report, force_alert=False):
                    self.send_telegram_alert(analysis_report, alert_type="UPDATE")
                    self.logger.info("📊 Regular update sent")
                    return
            
            # If no alerts sent but we have data, log the status
            self.logger.info("📝 Analysis complete - no alerts triggered")
                    
        except Exception as e:
            self.logger.error(f"Error in check_and_send_alerts: {e}")
    
    def setup_scheduler(self):
        """Setup scheduled tasks"""
        self.logger.info("Setting up scheduler...")
        
        # Data collection and analysis every 10 minutes
        schedule.every(settings.UPDATE_INTERVAL_MINUTES).minutes.do(
            self.run_full_cycle
        )
        
        # Send hourly summary (every 2 hours to avoid spam)
        schedule.every(2).hours.do(self.send_hourly_summary)
        
        # Model retraining every week
        schedule.every().sunday.at("02:00").do(self.retrain_model_weekly)
        
        # Initial data collection with historical data
        schedule.every().day.at("01:00").do(
            lambda: self.price_collector.collect_and_save(
                include_historical=True, 
                historical_days=7
            )
        )
        
        self.logger.info("Scheduler configured")
        self.logger.info(f"✅ Data collection every {settings.UPDATE_INTERVAL_MINUTES} minutes")
        self.logger.info("✅ Hourly summary every 2 hours")
        self.logger.info("✅ Critical alerts sent immediately")
        self.logger.info("✅ Regular alerts sent every 10 minutes (with data)")

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
    
    def send_telegram_alert(self, analysis_report, alert_type="UPDATE"):
        """Send enhanced Telegram alert with neural network analysis"""
        try:
            if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
                return False
                
            # Always use enhanced alert since we have advanced features integrated
            alert_message = create_enhanced_alert_message(analysis_report, alert_type)
            
            # Send to Telegram
            success = send_telegram_message(alert_message)
            
            if success:
                self.logger.info(f"📱 Telegram {alert_type} alert sent successfully")
                self.last_alert_time = datetime.utcnow()
            else:
                self.logger.warning("📱 Failed to send Telegram alert")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending Telegram alert: {e}")
            return False

    def should_send_alert(self, analysis_report, force_alert=False):
        """Determine if alert should be sent based on analysis"""
        try:
            if force_alert:
                return True
            
            # Check if this is a critical alert - always send critical alerts
            is_critical = self.is_critical_alert(analysis_report)
            if is_critical:
                # For critical alerts, only enforce a 2-minute minimum to prevent spam
                if self.last_alert_time:
                    time_since_last = datetime.utcnow() - self.last_alert_time
                    if time_since_last.total_seconds() < 120:  # 2 minutes minimum for critical
                        return False
                return True
            
            # For regular alerts, check if enough time has passed
            if self.last_alert_time:
                time_since_last = datetime.utcnow() - self.last_alert_time
                # Regular alerts every 10 minutes (matching our collection cycle)
                if time_since_last.total_seconds() < 600:  # 10 minutes for regular
                    return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in should_send_alert: {e}")
            return False

    def is_critical_alert(self, analysis_report):
        """Check if this is a critical alert that should be sent immediately"""
        try:
            # Critical price change > 5%
            if analysis_report.get('predictions'):
                predictions = analysis_report['predictions'].get('predictions', {})
                if '24h' in predictions:
                    change_pct = predictions['24h'].get('price_change_pct', 0)
                    if abs(change_pct) > 5:  # Critical threshold
                        return True
            
            # Critical sentiment (very negative or positive)
            if analysis_report.get('sentiment'):
                sentiment_score = analysis_report['sentiment']['overall']['overall_score']
                if sentiment_score < -0.7 or sentiment_score > 0.7:  # Very strong sentiment
                    return True
            
            # Strong technical signals
            if analysis_report.get('technical'):
                recommendation = analysis_report['technical'].get('recommendation', '')
                if recommendation in ['STRONG_BUY', 'STRONG_SELL']:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in is_critical_alert: {e}")
            return False

    def should_send_regular_update(self, analysis_report):
        """Check if we should send a regular update"""
        try:
            # Always send update if we haven't sent one in 30 minutes
            if self.last_alert_time:
                time_since_last = datetime.utcnow() - self.last_alert_time
                if time_since_last.total_seconds() >= 1800:  # 30 minutes
                    return True
            else:
                # First analysis - always send
                return True
            
            # Send for moderate price changes (2-5%)
            if analysis_report.get('predictions'):
                predictions = analysis_report['predictions'].get('predictions', {})
                if '24h' in predictions:
                    change_pct = predictions['24h'].get('price_change_pct', 0)
                    if abs(change_pct) > 2:  # 2-5% change
                        return True
            
            # Send for moderate sentiment changes
            if analysis_report.get('sentiment'):
                sentiment_score = analysis_report['sentiment']['overall']['overall_score']
                confidence = analysis_report['sentiment']['overall'].get('confidence', 0)
                # Lower threshold if we have high confidence
                threshold = 0.3 if confidence > 0.7 else 0.4
                if abs(sentiment_score) > threshold:
                    return True
            
            # Send for any notable technical signals
            if analysis_report.get('technical'):
                recommendation = analysis_report['technical'].get('recommendation', '')
                if recommendation in ['BUY', 'SELL', 'STRONG_BUY', 'STRONG_SELL']:
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in should_send_regular_update: {e}")
            return False

    def send_startup_message(self):
        """Send startup message to Telegram (only once per day)"""
        try:
            if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
                return False
            
            # Check if startup message was already sent today
            today = datetime.utcnow().date()
            startup_file = "last_startup.txt"
            
            # Check if we already sent startup message today
            try:
                with open(startup_file, 'r') as f:
                    last_startup = f.read().strip()
                    if last_startup == str(today):
                        self.logger.info("Startup message already sent today, skipping")
                        return False
            except FileNotFoundError:
                pass
                
            message = "🤖 **Bitcoin Analysis Bot Started!**\n\n"
            message += "✅ **Status:** Running 24/7\n"
            message += f"⏰ **Data Collection:** Every {settings.UPDATE_INTERVAL_MINUTES} minutes\n"
            message += f"� **Analysis Updates:** Every 30 minutes\n"
            message += f"📈 **Sources:** Twitter, Reddit, CoinGecko\n"
            message += f"� **ML Predictions:** {settings.PREDICTION_DAYS} days ahead\n\n"
            message += "**📱 Alert Types:**\n"
            message += "🚨 Critical: Price >5%, Strong signals\n"
            message += "📊 Regular: Price >2%, Sentiment changes\n"
            message += "⏰ Summary: Every 2 hours\n\n"
            message += "Ready to monitor Bitcoin markets! �"
            
            success = send_telegram_message(message)
            
            if success:
                # Save today's date to prevent spam
                with open(startup_file, 'w') as f:
                    f.write(str(today))
                self.logger.info("📱 Startup message sent to Telegram")
            else:
                self.logger.warning("📱 Failed to send startup message")
                
            return success
            
        except Exception as e:
            self.logger.error(f"Error sending startup message: {e}")
            return False

    def send_hourly_summary(self):
        """Send hourly summary with current status"""
        try:
            if not settings.TELEGRAM_BOT_TOKEN or not settings.TELEGRAM_CHAT_ID:
                return False
                
            # Check if we have recent data
            from database.models import get_session, PriceData, SentimentData
            session = get_session()
            
            try:
                # Get latest price
                latest_price = session.query(PriceData).order_by(PriceData.timestamp.desc()).first()
                
                # Get recent sentiment data (last 2 hours)
                cutoff_time = datetime.utcnow() - timedelta(hours=2)
                recent_sentiment = session.query(SentimentData).filter(
                    SentimentData.timestamp >= cutoff_time
                ).all()
                
                # Create summary message
                message = "📊 **Hourly Bitcoin Status**\n\n"
                
                if latest_price:
                    message += f"💰 **Current Price:** ${latest_price.price:,.2f}\n"
                else:
                    message += "💰 **Price:** No recent data\n"
                
                if recent_sentiment:
                    # Calculate average sentiment
                    scores = [s.sentiment_score for s in recent_sentiment]
                    avg_sentiment = sum(scores) / len(scores)
                    if avg_sentiment > 0.1:
                        emoji = "�"
                    elif avg_sentiment < -0.1:
                        emoji = "📉"
                    else:
                        emoji = "😐"
                    sent_text = f"💭 **Sentiment:** {emoji} {avg_sentiment:.3f}\n"
                    message += sent_text
                    message += f"📈 **Posts:** {len(recent_sentiment)}\n"
                else:
                    message += "💭 **Sentiment:** No recent data\n"
                    message += "📈 **Posts:** 0 (waiting for data)\n"
                
                # Add system status
                message += f"\n🔧 **Status:** Bot running normally\n"
                message += f"⏰ **Time:** {datetime.utcnow().strftime('%H:%M UTC')}"
                
                success = send_telegram_message(message)
                
                if success:
                    self.logger.info("📱 Hourly summary sent to Telegram")
                else:
                    self.logger.warning("📱 Failed to send hourly summary")
                    
                return success
                
            finally:
                session.close()
                
        except Exception as e:
            self.logger.error(f"Error sending hourly summary: {e}")
            return False

    def run(self):
        """Main run method"""
        self.logger.info("🚀 Bitcoin Analysis Bot Starting Up")
        
        # Initial setup
        self.setup_scheduler()
        
        # Start web server
        self.start_web_server()
        
        # Send startup message to Telegram
        self.send_startup_message()
        
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
