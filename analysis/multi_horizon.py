"""
Multi-Horizon Learning and Auto-Retraining for Bitcoin Prediction
Implements multi-timeframe analysis and automatic model retraining
"""

import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import schedule
import threading
import time

class MultiHorizonPredictor:
    """
    Multi-horizon predictor for 1h, 4h, and 24h timeframes
    Research shows multi-horizon approach improves prediction accuracy
    """
    
    def __init__(self, advanced_predictor=None):
        self.setup_logging()
        self.advanced_predictor = advanced_predictor
        
        # Different sequence lengths for different horizons
        self.horizons = {
            '1h': {'seq_length': 24, 'prediction_horizon': 1},   # 24 hours data, predict 1h ahead
            '4h': {'seq_length': 48, 'prediction_horizon': 4},   # 48 hours data, predict 4h ahead  
            '24h': {'seq_length': 168, 'prediction_horizon': 24}  # 7 days data, predict 24h ahead
        }
        
        self.models = {}  # Store models for each horizon
        self.last_predictions = {}  # Store last predictions for each horizon
        
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def train_multi_horizon_models(self, price_data, sentiment_data=None):
        """
        Train models for all time horizons
        
        Args:
            price_data: Historical price data
            sentiment_data: Historical sentiment data
            
        Returns:
            Dictionary with training results for each horizon
        """
        self.logger.info("🕐 Starting multi-horizon model training...")
        results = {}
        
        for horizon_name, config in self.horizons.items():
            try:
                self.logger.info(f"Training {horizon_name} model...")
                
                # Create predictor for this horizon
                from analysis.advanced_predictor import AdvancedCryptoPredictor
                predictor = AdvancedCryptoPredictor(
                    seq_length=config['seq_length'],
                    hidden_size=64,  # Smaller for faster training
                    num_layers=1,
                    dropout=0.2
                )
                
                # Engineer features with specific horizon
                features_data = predictor.engineer_features(
                    price_data=price_data,
                    sentiment_data=sentiment_data
                )
                
                if features_data is not None and not features_data.empty:
                    # Modify target creation for specific horizon
                    features_data['target'] = self._create_horizon_target(
                        price_data, 
                        horizon=config['prediction_horizon']
                    )
                    
                    # Train model
                    success = predictor.train_model(
                        features_data=features_data,
                        epochs=30,  # Faster training
                        batch_size=16,
                        learning_rate=0.001
                    )
                    
                    if success:
                        self.models[horizon_name] = predictor
                        results[horizon_name] = {
                            'status': 'success',
                            'model_trained': True,
                            'data_points': len(features_data)
                        }
                        self.logger.info(f"✅ {horizon_name} model trained successfully")
                    else:
                        results[horizon_name] = {
                            'status': 'failed',
                            'error': 'Training failed'
                        }
                        self.logger.error(f"❌ {horizon_name} model training failed")
                else:
                    results[horizon_name] = {
                        'status': 'failed',
                        'error': 'Feature engineering failed'
                    }
                    
            except Exception as e:
                self.logger.error(f"Error training {horizon_name} model: {e}")
                results[horizon_name] = {
                    'status': 'failed',
                    'error': str(e)
                }
        
        return results
    
    def _create_horizon_target(self, price_data, horizon=24, threshold=0.02):
        """
        Create target variable for specific horizon
        
        Args:
            price_data: Price data
            horizon: Hours ahead to predict
            threshold: Minimum change threshold
            
        Returns:
            Target array
        """
        try:
            if isinstance(price_data, pd.DataFrame):
                prices = price_data['close']
            else:
                prices = price_data
            
            future_prices = prices.shift(-horizon)
            returns = (future_prices - prices) / prices
            
            target = np.where(returns > threshold, 2,  # UP
                     np.where(returns < -threshold, 0, 1))  # DOWN, HOLD
            
            return target
            
        except Exception as e:
            self.logger.error(f"Error creating horizon target: {e}")
            return np.ones(len(price_data))  # Default to HOLD
    
    def predict_all_horizons(self, recent_data):
        """
        Make predictions for all time horizons
        
        Args:
            recent_data: Recent price and sentiment data
            
        Returns:
            Dictionary with predictions for each horizon
        """
        try:
            predictions = {}
            
            for horizon_name, model in self.models.items():
                try:
                    if model is not None:
                        # Engineer features for this horizon
                        features_data = model.engineer_features(
                            price_data=recent_data.get('price_data'),
                            sentiment_data=recent_data.get('sentiment_data')
                        )
                        
                        if features_data is not None and not features_data.empty:
                            # Remove target column if present
                            feature_cols = [col for col in features_data.columns if col != 'target']
                            prediction_data = features_data[feature_cols]
                            
                            # Make prediction
                            prediction = model.predict(prediction_data, return_attention=True)
                            
                            if prediction:
                                predictions[horizon_name] = {
                                    'prediction': prediction,
                                    'horizon_hours': self.horizons[horizon_name]['prediction_horizon'],
                                    'timestamp': datetime.utcnow(),
                                    'confidence_level': self._classify_confidence(prediction['confidence'])
                                }
                                
                                self.logger.info(
                                    f"{horizon_name} prediction: {prediction['direction_text']} "
                                    f"(confidence: {prediction['confidence']:.2f})"
                                )
                            else:
                                predictions[horizon_name] = {
                                    'error': 'Prediction failed',
                                    'horizon_hours': self.horizons[horizon_name]['prediction_horizon']
                                }
                        else:
                            predictions[horizon_name] = {
                                'error': 'Feature engineering failed',
                                'horizon_hours': self.horizons[horizon_name]['prediction_horizon']
                            }
                    else:
                        predictions[horizon_name] = {
                            'error': 'Model not trained',
                            'horizon_hours': self.horizons[horizon_name]['prediction_horizon']
                        }
                        
                except Exception as e:
                    self.logger.error(f"Error predicting {horizon_name}: {e}")
                    predictions[horizon_name] = {
                        'error': str(e),
                        'horizon_hours': self.horizons[horizon_name]['prediction_horizon']
                    }
            
            # Store last predictions
            self.last_predictions = predictions
            return predictions
            
        except Exception as e:
            self.logger.error(f"Error in multi-horizon prediction: {e}")
            return {}
    
    def _classify_confidence(self, confidence):
        """Classify confidence level"""
        if confidence >= 0.8:
            return "Very High"
        elif confidence >= 0.6:
            return "High"
        elif confidence >= 0.4:
            return "Medium"
        else:
            return "Low"
    
    def get_consensus_prediction(self, predictions=None):
        """
        Get consensus prediction from all horizons
        
        Args:
            predictions: Multi-horizon predictions (uses last if None)
            
        Returns:
            Dictionary with consensus prediction
        """
        try:
            if predictions is None:
                predictions = self.last_predictions
            
            if not predictions:
                return {'consensus': 'HOLD', 'confidence': 0.0, 'agreement': 0.0}
            
            # Extract valid predictions
            valid_predictions = []
            for horizon, pred_data in predictions.items():
                if 'prediction' in pred_data and 'error' not in pred_data:
                    pred = pred_data['prediction']
                    valid_predictions.append({
                        'horizon': horizon,
                        'direction': pred['direction'],
                        'confidence': pred['confidence'],
                        'direction_text': pred['direction_text']
                    })
            
            if not valid_predictions:
                return {'consensus': 'HOLD', 'confidence': 0.0, 'agreement': 0.0}
            
            # Calculate consensus
            directions = [p['direction'] for p in valid_predictions]
            confidences = [p['confidence'] for p in valid_predictions]
            
            # Most common direction
            from collections import Counter
            direction_counts = Counter(directions)
            consensus_direction = direction_counts.most_common(1)[0][0]
            
            # Agreement level
            agreement = direction_counts[consensus_direction] / len(directions)
            
            # Average confidence for consensus direction
            consensus_confidences = [
                p['confidence'] for p in valid_predictions 
                if p['direction'] == consensus_direction
            ]
            avg_confidence = np.mean(consensus_confidences)
            
            # Convert direction to text
            direction_map = {0: 'DOWN', 1: 'HOLD', 2: 'UP'}
            consensus_text = direction_map.get(consensus_direction, 'HOLD')
            
            return {
                'consensus': consensus_text,
                'consensus_direction': consensus_direction,
                'confidence': avg_confidence,
                'agreement': agreement,
                'num_models': len(valid_predictions),
                'individual_predictions': valid_predictions
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating consensus: {e}")
            return {'consensus': 'HOLD', 'confidence': 0.0, 'agreement': 0.0}

class AutoRetrainingManager:
    """
    Manages automatic retraining of models every 24-48 hours
    """
    
    def __init__(self, bot_instance=None):
        self.setup_logging()
        self.bot_instance = bot_instance
        self.multi_horizon_predictor = None
        self.last_retrain_time = None
        self.retrain_interval_hours = 48  # Retrain every 48 hours
        self.is_running = False
        
    def setup_logging(self):
        """Setup logging"""
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.INFO)
    
    def start_auto_retraining(self):
        """Start automatic retraining scheduler"""
        try:
            self.logger.info("🔄 Starting auto-retraining scheduler...")
            
            # Schedule retraining every 48 hours
            schedule.every(self.retrain_interval_hours).hours.do(self._retrain_all_models)
            
            # Also schedule for specific times (less load on server)
            schedule.every().day.at("02:00").do(self._retrain_if_needed)
            schedule.every().day.at("14:00").do(self._retrain_if_needed)
            
            # Start scheduler in background thread
            self.is_running = True
            scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
            scheduler_thread.start()
            
            self.logger.info(f"✅ Auto-retraining scheduled every {self.retrain_interval_hours} hours")
            
        except Exception as e:
            self.logger.error(f"Error starting auto-retraining: {e}")
    
    def _run_scheduler(self):
        """Run the scheduler in background"""
        while self.is_running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(300)  # Wait 5 minutes on error
    
    def _retrain_if_needed(self):
        """Retrain models if enough time has passed"""
        try:
            current_time = datetime.utcnow()
            
            # Check if we need to retrain
            if (self.last_retrain_time is None or 
                (current_time - self.last_retrain_time).total_seconds() > self.retrain_interval_hours * 3600):
                
                self.logger.info("⏰ Auto-retraining time reached")
                self._retrain_all_models()
            else:
                hours_until_retrain = self.retrain_interval_hours - (current_time - self.last_retrain_time).total_seconds() / 3600
                self.logger.info(f"⏳ Next retrain in {hours_until_retrain:.1f} hours")
                
        except Exception as e:
            self.logger.error(f"Error checking retrain schedule: {e}")
    
    def _retrain_all_models(self):
        """Retrain all models with fresh data"""
        try:
            self.logger.info("🔄 Starting automatic model retraining...")
            
            if not self.bot_instance:
                self.logger.error("Bot instance not available for retraining")
                return
            
            # Get fresh data for training
            recent_prices = self.bot_instance.price_collector.get_recent_prices(days=90)
            recent_sentiment = None
            
            if hasattr(self.bot_instance, 'sentiment_analyzer'):
                recent_sentiment = self.bot_instance.sentiment_analyzer.get_recent_sentiment_dataframe(
                    hours=90*24
                )
            
            if recent_prices is None or len(recent_prices) < 100:
                self.logger.warning("Not enough data for retraining")
                return
            
            # Retrain main advanced predictor
            if hasattr(self.bot_instance, 'advanced_predictor') and self.bot_instance.advanced_predictor:
                self.logger.info("Retraining main advanced predictor...")
                success = self.bot_instance.train_advanced_model()
                if success:
                    self.logger.info("✅ Main advanced predictor retrained")
                else:
                    self.logger.error("❌ Main advanced predictor retraining failed")
            
            # Retrain LightGBM if available
            if hasattr(self.bot_instance, 'lightgbm_predictor') and self.bot_instance.lightgbm_predictor:
                self.logger.info("Retraining LightGBM predictor...")
                try:
                    features_df = self.bot_instance.lightgbm_predictor.prepare_features(
                        recent_prices, recent_sentiment
                    )
                    if features_df is not None and not features_df.empty:
                        # Retrain LightGBM (if method exists)
                        if hasattr(self.bot_instance.lightgbm_predictor, 'retrain_model'):
                            success = self.bot_instance.lightgbm_predictor.retrain_model(features_df)
                            if success:
                                self.logger.info("✅ LightGBM predictor retrained")
                            else:
                                self.logger.error("❌ LightGBM predictor retraining failed")
                except Exception as e:
                    self.logger.error(f"LightGBM retraining error: {e}")
            
            # Retrain multi-horizon models if available
            if self.multi_horizon_predictor:
                self.logger.info("Retraining multi-horizon models...")
                results = self.multi_horizon_predictor.train_multi_horizon_models(
                    recent_prices, recent_sentiment
                )
                
                success_count = sum(1 for r in results.values() if r.get('status') == 'success')
                self.logger.info(f"✅ {success_count}/{len(results)} multi-horizon models retrained")
            
            # Update last retrain time
            self.last_retrain_time = datetime.utcnow()
            
            self.logger.info("🎯 Automatic retraining completed")
            
            # Send notification if possible
            if hasattr(self.bot_instance, 'send_telegram_message'):
                try:
                    from utils.helpers import send_telegram_message
                    send_telegram_message(
                        f"🔄 Models automatically retrained at {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}\n"
                        f"✅ Fresh data incorporated for improved predictions"
                    )
                except:
                    pass  # Don't fail if notification fails
                    
        except Exception as e:
            self.logger.error(f"Error in automatic retraining: {e}")
            import traceback
            traceback.print_exc()
    
    def stop_auto_retraining(self):
        """Stop automatic retraining"""
        self.is_running = False
        schedule.clear()
        self.logger.info("🛑 Auto-retraining stopped")

def setup_multi_horizon_system(bot_instance):
    """
    Setup multi-horizon prediction and auto-retraining system
    
    Args:
        bot_instance: Main bot instance
        
    Returns:
        Tuple of (MultiHorizonPredictor, AutoRetrainingManager)
    """
    try:
        # Initialize multi-horizon predictor
        multi_horizon = MultiHorizonPredictor()
        
        # Initialize auto-retraining manager
        auto_retrain = AutoRetrainingManager(bot_instance)
        auto_retrain.multi_horizon_predictor = multi_horizon
        
        # Start auto-retraining
        auto_retrain.start_auto_retraining()
        
        return multi_horizon, auto_retrain
        
    except Exception as e:
        logging.error(f"Error setting up multi-horizon system: {e}")
        return None, None
