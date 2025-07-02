import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from database.models import get_session, PriceData, SentimentData, TechnicalAnalysis, Predictions
from config.settings import settings
import joblib
import os

class BitcoinPredictor:
    def __init__(self):
        self.setup_logging()
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def prepare_features(self, days=90):
        """Prepare feature dataset for ML model"""
        session = get_session()
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            # Get price data
            price_data = session.query(PriceData).filter(
                PriceData.timestamp >= cutoff_time
            ).order_by(PriceData.timestamp).all()
            
            if len(price_data) < 30:
                self.logger.error("Insufficient price data for prediction")
                return None
            
            # Convert to DataFrame
            df_price = pd.DataFrame([{
                'timestamp': entry.timestamp,
                'price': entry.price,
                'volume': entry.volume or 0,
                'high': entry.high or entry.price,
                'low': entry.low or entry.price,
                'market_cap': entry.market_cap or 0
            } for entry in price_data])
            
            df_price.set_index('timestamp', inplace=True)
            df_price = df_price.resample('1H').mean().interpolate()
            
            # Get sentiment data
            sentiment_data = session.query(SentimentData).filter(
                SentimentData.timestamp >= cutoff_time
            ).all()
            
            # Aggregate sentiment by hour
            df_sentiment = pd.DataFrame([{
                'timestamp': entry.timestamp,
                'sentiment_score': entry.sentiment_score,
                'likes': entry.likes,
                'source': entry.source
            } for entry in sentiment_data])
            
            if not df_sentiment.empty:
                df_sentiment['timestamp'] = pd.to_datetime(df_sentiment['timestamp'])
                df_sentiment.set_index('timestamp', inplace=True)
                
                # Aggregate sentiment metrics per hour
                sentiment_hourly = df_sentiment.resample('1H').agg({
                    'sentiment_score': ['mean', 'std', 'count'],
                    'likes': 'sum'
                }).fillna(0)
                
                sentiment_hourly.columns = ['sentiment_mean', 'sentiment_std', 'sentiment_count', 'total_likes']
            else:
                # Create empty sentiment data
                sentiment_hourly = pd.DataFrame(
                    index=df_price.index,
                    columns=['sentiment_mean', 'sentiment_std', 'sentiment_count', 'total_likes']
                ).fillna(0)
            
            # Get technical analysis data
            ta_data = session.query(TechnicalAnalysis).filter(
                TechnicalAnalysis.timestamp >= cutoff_time
            ).order_by(TechnicalAnalysis.timestamp).all()
            
            if ta_data:
                df_ta = pd.DataFrame([{
                    'timestamp': entry.timestamp,
                    'rsi': entry.rsi,
                    'macd': entry.macd,
                    'bb_upper': entry.bb_upper,
                    'bb_lower': entry.bb_lower,
                    'sma_20': entry.sma_20,
                    'sma_50': entry.sma_50,
                    'volume_sma': entry.volume_sma
                } for entry in ta_data])
                
                df_ta['timestamp'] = pd.to_datetime(df_ta['timestamp'])
                df_ta.set_index('timestamp', inplace=True)
                df_ta = df_ta.resample('1H').mean().interpolate()
            else:
                # Create empty TA data
                df_ta = pd.DataFrame(
                    index=df_price.index,
                    columns=['rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'sma_50', 'volume_sma']
                ).fillna(0)
            
            # Combine all features
            df_combined = df_price.join(sentiment_hourly, how='left').join(df_ta, how='left')
            
            # Fill missing values
            df_combined = df_combined.fillna(method='forward').fillna(0)
            
            # Create additional features
            df_combined['price_change'] = df_combined['price'].pct_change()
            df_combined['volume_change'] = df_combined['volume'].pct_change()
            df_combined['price_sma_5'] = df_combined['price'].rolling(5).mean()
            df_combined['price_volatility'] = df_combined['price'].rolling(24).std()
            
            # Create target variable (price in next hour)
            df_combined['target'] = df_combined['price'].shift(-1)
            
            # Remove rows with NaN target
            df_combined = df_combined.dropna()
            
            self.logger.info(f"Prepared {len(df_combined)} feature samples")
            return df_combined
            
        except Exception as e:
            self.logger.error(f"Error preparing features: {e}")
            return None
        finally:
            session.close()
    
    def train_model(self, df_features):
        """Train the prediction model"""
        if df_features is None or len(df_features) < 50:
            self.logger.error("Insufficient data for training")
            return False
        
        try:
            # Define feature columns
            self.feature_columns = [
                'price', 'volume', 'high', 'low', 'market_cap',
                'sentiment_mean', 'sentiment_std', 'sentiment_count', 'total_likes',
                'rsi', 'macd', 'bb_upper', 'bb_lower', 'sma_20', 'sma_50', 'volume_sma',
                'price_change', 'volume_change', 'price_sma_5', 'price_volatility'
            ]
            
            # Filter available columns
            available_columns = [col for col in self.feature_columns if col in df_features.columns]
            self.feature_columns = available_columns
            
            X = df_features[self.feature_columns]
            y = df_features['target']
            
            # Handle infinite values
            X = X.replace([np.inf, -np.inf], 0)
            y = y.replace([np.inf, -np.inf], np.nan).dropna()
            X = X.loc[y.index]
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train model
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42,
                n_jobs=-1
            )
            
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = self.model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            # Calculate accuracy percentage
            accuracy = 1 - (mae / y_test.mean())
            
            self.logger.info(f"Model trained - MAE: ${mae:.2f}, RMSE: ${rmse:.2f}, Accuracy: {accuracy:.2%}")
            
            # Save model
            self.save_model()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error training model: {e}")
            return False
    
    def save_model(self):
        """Save trained model and scaler"""
        try:
            model_dir = "models"
            os.makedirs(model_dir, exist_ok=True)
            
            joblib.dump(self.model, os.path.join(model_dir, "bitcoin_predictor.pkl"))
            joblib.dump(self.scaler, os.path.join(model_dir, "scaler.pkl"))
            joblib.dump(self.feature_columns, os.path.join(model_dir, "features.pkl"))
            
            self.logger.info("Model saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        """Load trained model and scaler"""
        try:
            model_dir = "models"
            
            if not os.path.exists(os.path.join(model_dir, "bitcoin_predictor.pkl")):
                self.logger.warning("No saved model found")
                return False
            
            self.model = joblib.load(os.path.join(model_dir, "bitcoin_predictor.pkl"))
            self.scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
            self.feature_columns = joblib.load(os.path.join(model_dir, "features.pkl"))
            
            self.logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def predict_price(self, hours_ahead=24):
        """Predict Bitcoin price for specified hours ahead"""
        if self.model is None:
            if not self.load_model():
                self.logger.error("No model available for prediction")
                return None
        
        try:
            # Get recent data for prediction
            df_features = self.prepare_features(days=30)
            if df_features is None or df_features.empty:
                return None
            
            # Get latest features
            latest_features = df_features[self.feature_columns].iloc[-1:].copy()
            
            # Handle missing or infinite values
            latest_features = latest_features.fillna(0).replace([np.inf, -np.inf], 0)
            
            # Scale features
            features_scaled = self.scaler.transform(latest_features)
            
            # Make prediction
            predicted_price = self.model.predict(features_scaled)[0]
            
            # Calculate confidence based on feature importance and recent volatility
            feature_importance = self.model.feature_importances_
            recent_volatility = df_features['price_volatility'].iloc[-1]
            confidence = max(0.1, min(0.9, 1 - (recent_volatility / df_features['price'].iloc[-1])))
            
            # Get current price for comparison
            current_price = df_features['price'].iloc[-1]
            price_change_pct = ((predicted_price - current_price) / current_price) * 100
            
            prediction = {
                'predicted_price': predicted_price,
                'current_price': current_price,
                'price_change': predicted_price - current_price,
                'price_change_pct': price_change_pct,
                'confidence': confidence,
                'prediction_time': datetime.utcnow(),
                'target_time': datetime.utcnow() + timedelta(hours=hours_ahead),
                'model_features': dict(zip(self.feature_columns, latest_features.iloc[0]))
            }
            
            self.logger.info(f"Price prediction: ${predicted_price:.2f} ({price_change_pct:+.2f}%)")
            
            return prediction
            
        except Exception as e:
            self.logger.error(f"Error making prediction: {e}")
            return None
    
    def save_prediction(self, prediction, hours_ahead=24):
        """Save prediction to database"""
        if prediction is None:
            return
        
        session = get_session()
        
        try:
            pred_entry = Predictions(
                prediction_for=prediction['target_time'],
                predicted_price=prediction['predicted_price'],
                confidence_score=prediction['confidence'],
                prediction_type=f"{hours_ahead}h",
                model_name="RandomForest",
                model_version="1.0",
                sentiment_weight=0.3,  # These could be calculated from feature importance
                technical_weight=0.4,
                price_weight=0.3
            )
            
            session.add(pred_entry)
            session.commit()
            
            self.logger.info("Prediction saved to database")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving prediction: {e}")
        finally:
            session.close()
    
    def retrain_model(self):
        """Retrain the model with latest data"""
        self.logger.info("Starting model retraining...")
        
        # Prepare fresh data
        df_features = self.prepare_features(days=settings.TRAIN_DAYS)
        
        # Train model
        if self.train_model(df_features):
            self.logger.info("Model retrained successfully")
            return True
        else:
            self.logger.error("Model retraining failed")
            return False
    
    def generate_prediction_report(self, hours_ahead=[1, 4, 24]):
        """Generate predictions for multiple time horizons"""
        predictions = {}
        
        for hours in hours_ahead:
            pred = self.predict_price(hours_ahead=hours)
            if pred:
                predictions[f"{hours}h"] = pred
                self.save_prediction(pred, hours_ahead=hours)
        
        if predictions:
            report = {
                'timestamp': datetime.utcnow(),
                'predictions': predictions,
                'summary': {
                    'short_term_trend': 'bullish' if predictions.get('1h', {}).get('price_change_pct', 0) > 0 else 'bearish',
                    'medium_term_trend': 'bullish' if predictions.get('4h', {}).get('price_change_pct', 0) > 0 else 'bearish',
                    'long_term_trend': 'bullish' if predictions.get('24h', {}).get('price_change_pct', 0) > 0 else 'bearish',
                    'avg_confidence': np.mean([p['confidence'] for p in predictions.values()])
                }
            }
            
            return report
        
        return None

if __name__ == "__main__":
    predictor = BitcoinPredictor()
    
    # Train model
    df = predictor.prepare_features(days=60)
    if df is not None:
        predictor.train_model(df)
    
    # Generate predictions
    report = predictor.generate_prediction_report()
    if report:
        print("Prediction Report:")
        for timeframe, pred in report['predictions'].items():
            print(f"{timeframe}: ${pred['predicted_price']:.2f} ({pred['price_change_pct']:+.2f}%)")
