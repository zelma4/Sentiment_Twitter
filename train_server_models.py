#!/usr/bin/env python3
"""
Server Model Training Script
Optimized for running on production server with real data
"""

import os
import sys
import logging
import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def check_database_data():
    """Check available data in the database"""
    try:
        conn = sqlite3.connect('bitcoin_sentiment.db')
        
        # Check price data
        price_df = pd.read_sql_query("""
            SELECT COUNT(*) as count, 
                   MIN(timestamp) as earliest, 
                   MAX(timestamp) as latest 
            FROM price_data
        """, conn)
        
        # Check sentiment data
        sentiment_df = pd.read_sql_query("""
            SELECT COUNT(*) as count,
                   MIN(timestamp) as earliest,
                   MAX(timestamp) as latest
            FROM sentiment_data  
        """, conn)
        
        conn.close()
        
        logger.info(f"📊 Price data: {price_df.iloc[0]['count']} records")
        logger.info(f"   Range: {price_df.iloc[0]['earliest']} to {price_df.iloc[0]['latest']}")
        
        logger.info(f"💭 Sentiment data: {sentiment_df.iloc[0]['count']} records")
        logger.info(f"   Range: {sentiment_df.iloc[0]['earliest']} to {sentiment_df.iloc[0]['latest']}")
        
        return price_df.iloc[0]['count'], sentiment_df.iloc[0]['count']
        
    except Exception as e:
        logger.error(f"Database check failed: {e}")
        return 0, 0

def load_real_data():
    """Load real data from database"""
    try:
        conn = sqlite3.connect('bitcoin_sentiment.db')
        
        # Load price data
        price_query = """
            SELECT timestamp, open, high, low, close, volume
            FROM price_data 
            ORDER BY timestamp DESC 
            LIMIT 2000
        """
        price_data = pd.read_sql_query(price_query, conn)
        price_data['timestamp'] = pd.to_datetime(price_data['timestamp'])
        price_data = price_data.sort_values('timestamp')
        
        # Load sentiment data
        sentiment_query = """
            SELECT timestamp, sentiment_score, compound, pos, neu, neg, source, content
            FROM sentiment_data 
            ORDER BY timestamp DESC 
            LIMIT 5000
        """
        sentiment_data = pd.read_sql_query(sentiment_query, conn)
        sentiment_data['timestamp'] = pd.to_datetime(sentiment_data['timestamp'])
        sentiment_data = sentiment_data.sort_values('timestamp')
        
        conn.close()
        
        logger.info(f"✅ Loaded {len(price_data)} price records and {len(sentiment_data)} sentiment records")
        return price_data, sentiment_data
        
    except Exception as e:
        logger.error(f"Failed to load real data: {e}")
        return None, None

def train_lightgbm_with_real_data():
    """Train LightGBM with real server data"""
    try:
        from analysis.lightgbm_predictor import LightGBMPredictor
        
        logger.info("🚀 Training LightGBM with real server data...")
        
        # Load real data
        price_data, sentiment_data = load_real_data()
        
        if price_data is None or len(price_data) < 50:
            logger.error("Insufficient price data for LightGBM training")
            return False
        
        # Initialize predictor
        predictor = LightGBMPredictor()
        
        # Prepare features
        features_df = predictor.prepare_features(price_data, sentiment_data)
        
        if features_df.empty:
            logger.error("Failed to prepare features")
            return False
        
        logger.info(f"📊 Training with {len(features_df)} feature records")
        
        # Train model
        result = predictor.train_model(features_df)
        
        if result and result.get('success'):
            logger.info(f"✅ LightGBM training successful! Accuracy: {result.get('accuracy', 'N/A')}")
            return True
        else:
            logger.error("❌ LightGBM training failed")
            return False
            
    except Exception as e:
        logger.error(f"LightGBM training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def train_advanced_model_with_real_data():
    """Train Advanced Neural Network with real server data"""
    try:
        from analysis.advanced_predictor import AdvancedCryptoPredictor
        
        logger.info("🧠 Training Advanced Neural Network with real server data...")
        
        # Load real data
        price_data, sentiment_data = load_real_data()
        
        if price_data is None or len(price_data) < 100:
            logger.error("Insufficient data for Advanced Neural Network training")
            return False
        
        # Initialize predictor
        predictor = AdvancedCryptoPredictor()
        
        # Engineer features
        features_data = predictor.engineer_features(
            price_data=price_data,
            sentiment_data=sentiment_data
        )
        
        if features_data is None or features_data.empty:
            logger.error("Failed to engineer features")
            return False
        
        logger.info(f"📊 Training with {len(features_data)} engineered feature records")
        
        # Train model with appropriate parameters for server
        success = predictor.train_model(
            features_data=features_data,
            epochs=20,  # Reasonable for server training
            batch_size=16,
            learning_rate=0.001
        )
        
        if success:
            logger.info("✅ Advanced Neural Network training successful!")
            return True
        else:
            logger.error("❌ Advanced Neural Network training failed")
            return False
            
    except Exception as e:
        logger.error(f"Advanced Neural Network training error: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_predictions():
    """Test predictions with trained models"""
    try:
        logger.info("🧪 Testing predictions with trained models...")
        
        # Test LightGBM
        try:
            from analysis.lightgbm_predictor import LightGBMPredictor
            lgb_predictor = LightGBMPredictor()
            
            # Get recent data for testing
            price_data, sentiment_data = load_real_data()
            
            if price_data is not None and len(price_data) > 10:
                prediction = lgb_predictor.predict_next_direction(
                    price_data=price_data.tail(100),
                    sentiment_data=sentiment_data.tail(200) if sentiment_data is not None else None
                )
                
                if prediction:
                    logger.info(f"✅ LightGBM Test: {prediction['direction_text']} (confidence: {prediction['confidence']:.3f})")
                else:
                    logger.warning("⚠️ LightGBM prediction test failed")
            
        except Exception as e:
            logger.error(f"LightGBM test failed: {e}")
        
        # Test Advanced Neural Network
        try:
            from analysis.advanced_predictor import AdvancedCryptoPredictor
            adv_predictor = AdvancedCryptoPredictor()
            
            # Load model if exists
            if os.path.exists('models/advanced_crypto_model.pth'):
                if adv_predictor.load_model():
                    logger.info("✅ Advanced model loaded successfully")
                    
                    # Try a prediction
                    price_data, sentiment_data = load_real_data()
                    if price_data is not None and len(price_data) > 60:
                        features_data = adv_predictor.engineer_features(
                            price_data=price_data.tail(100),
                            sentiment_data=sentiment_data.tail(200) if sentiment_data is not None else None
                        )
                        
                        if features_data is not None and not features_data.empty:
                            # Remove target if present
                            test_features = features_data.drop(columns=['target'] if 'target' in features_data.columns else [])
                            
                            prediction = adv_predictor.predict(test_features.tail(1))
                            
                            if prediction:
                                logger.info(f"✅ Advanced Neural Test: {prediction['direction_text']} (confidence: {prediction['confidence']:.3f})")
                            else:
                                logger.warning("⚠️ Advanced Neural prediction test failed")
                else:
                    logger.warning("⚠️ Failed to load Advanced model")
            else:
                logger.info("ℹ️ Advanced model not found - training required")
        
        except Exception as e:
            logger.error(f"Advanced Neural test failed: {e}")
            
    except Exception as e:
        logger.error(f"Prediction testing failed: {e}")

def main():
    """Main training function"""
    logger.info("🚀 Server Model Training Started")
    logger.info("================================")
    
    # Check available data
    price_count, sentiment_count = check_database_data()
    
    if price_count < 10:
        logger.error("❌ Insufficient data for training. Need at least 10 price records.")
        logger.info("🔄 Bot needs to run longer to collect more data")
        return
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    success_count = 0
    
    # Train LightGBM
    if price_count >= 50:
        logger.info("\n1️⃣ Training LightGBM Model...")
        if train_lightgbm_with_real_data():
            success_count += 1
        else:
            logger.warning("LightGBM training skipped due to errors")
    else:
        logger.warning(f"Skipping LightGBM - need 50+ records, have {price_count}")
    
    # Train Advanced Neural Network  
    if price_count >= 100:
        logger.info("\n2️⃣ Training Advanced Neural Network...")
        if train_advanced_model_with_real_data():
            success_count += 1
        else:
            logger.warning("Advanced Neural Network training skipped due to errors")
    else:
        logger.warning(f"Skipping Advanced Neural - need 100+ records, have {price_count}")
    
    # Test predictions
    logger.info("\n3️⃣ Testing Predictions...")
    test_predictions()
    
    # Final summary
    logger.info(f"\n✅ Training Complete!")
    logger.info(f"📊 Successfully trained: {success_count}/2 models")
    logger.info(f"💾 Data available: {price_count} price, {sentiment_count} sentiment records")
    
    if success_count > 0:
        logger.info("🎯 Models are ready for production use!")
    else:
        logger.info("⏳ Models will be trained automatically as more data becomes available")
    
    logger.info("🔄 For automatic retraining, models retrain weekly on Sundays at 2 AM")

if __name__ == "__main__":
    main()
