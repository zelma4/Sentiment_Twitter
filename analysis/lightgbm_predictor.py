"""
LightGBM Bitcoin Price Direction Predictor

Advanced machine learning model for predicting Bitcoin price movements
using technical indicators and sentiment analysis.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta

try:
    import lightgbm as lgb
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)


class LightGBMPredictor:
    """
    LightGBM-based Bitcoin price direction predictor
    """
    
    def __init__(self):
        """Initialize the predictor"""
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.is_trained = False
        
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available. Install with: pip install lightgbm")
    
    def prepare_features(self, 
                        price_data: pd.DataFrame, 
                        sentiment_data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for the model
        
        Args:
            price_data: DataFrame with OHLCV data
            sentiment_data: DataFrame with sentiment scores
            
        Returns:
            DataFrame with engineered features
        """
        # Validate input data
        if price_data is None or price_data.empty:
            logger.warning("Empty or None price data provided")
            return pd.DataFrame()
        
        # Check required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in price_data.columns]
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return pd.DataFrame()
        
        # Check minimum data requirements
        if len(price_data) < 10:  # Reduced from 50 to 10 for practical use
            logger.warning(f"Insufficient data: {len(price_data)} rows (need at least 10)")
            return pd.DataFrame()
        
        df = price_data.copy()
        
        # Ensure we have a date index or column
        if df.index.name != 'date' and 'date' not in df.columns:
            # Create a date column if none exists
            df['date'] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            df.set_index('date', inplace=True)
        elif 'date' in df.columns:
            df.set_index('date', inplace=True)
        
        # Technical indicators (adjusted for small datasets)
        df['roc_1d'] = df['close'].pct_change(1)
        df['roc_3d'] = df['close'].pct_change(min(3, len(df)-1))
        df['high_low_range'] = (df['high'] - df['low']) / df['close']
        
        # Moving averages (use smaller windows for limited data)
        sma_window = min(5, len(df)-1) if len(df) < 10 else 10
        df['sma_10'] = df['close'].rolling(sma_window).mean()
        df['close_to_sma10_ratio'] = df['close'] / df['sma_10']
        
        # Bollinger Bands (use smaller window for limited data)
        bb_period = min(10, len(df)-1) if len(df) < 20 else 20
        df['bb_middle'] = df['close'].rolling(bb_period).mean()
        bb_std = df['close'].rolling(bb_period).std()
        df['bb_upper'] = df['bb_middle'] + (bb_std * 2)
        df['bb_lower'] = df['bb_middle'] - (bb_std * 2)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        # Volume indicators
        vol_window = min(5, len(df)-1) if len(df) < 10 else 10
        df['volume_ma'] = df['volume'].rolling(vol_window).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Merge sentiment data
        if sentiment_data is not None and not sentiment_data.empty and 'sentiment' in sentiment_data.columns:
            # Ensure sentiment data has proper index
            if sentiment_data.index.name != 'date' and 'date' in sentiment_data.columns:
                sentiment_data = sentiment_data.set_index('date')
            
            df = df.merge(sentiment_data, left_index=True, right_index=True, how='left')
            
            # Fill NaN values in sentiment with 0
            df.loc[:, 'sentiment'] = df['sentiment'].fillna(0)
            
            # Sentiment features (adjusted for small datasets)
            sent_window = min(3, len(df)-1) if len(df) < 5 else 5
            df['sent_5d'] = df['sentiment'].rolling(sent_window).mean()
            df['sent_delta'] = df['sentiment'].diff()
            df['sent_accel'] = df['sent_delta'].diff()
            df['sent_vol'] = df['sentiment'].rolling(sent_window).std()
            
            # Sentiment quantiles (handle edge cases)
            try:
                df['sent_q'] = pd.qcut(df['sentiment'], q=5, labels=False, duplicates='drop')
            except ValueError:
                # If not enough unique values for quantiles, use simple thresholds
                df['sent_q'] = pd.cut(df['sentiment'], bins=5, labels=False)
                
            df.loc[:, 'sent_q'] = df['sent_q'].fillna(2)  # Fill with middle quantile
            df['sent_q2_flag'] = (df['sent_q'] == 1).astype(int)
            df['sent_q5_flag'] = (df['sent_q'] == 4).astype(int)
            
            # Sentiment crosses
            df['sent_cross_up'] = ((df['sentiment'] > df['sent_5d']) & 
                                  (df['sentiment'] > 0)).astype(int)
            df['sent_neg'] = (df['sentiment'] < -0.2).astype(int)
            
            # Interaction features
            df['sent_q2_flag_x_close_to_sma10'] = (df['sent_q2_flag'] * 
                                                  df['close_to_sma10_ratio'])
            df['sent_cross_up_x_high_low_range'] = (df['sent_cross_up'] * 
                                                   df['high_low_range'])
            df['sent_neg_x_high_low_range'] = (df['sent_neg'] * 
                                              df['high_low_range'])
        else:
            # Fill with neutral values if no sentiment data
            sentiment_features = ['sentiment', 'sent_5d', 'sent_delta', 'sent_accel', 
                                'sent_vol', 'sent_q', 'sent_q2_flag', 'sent_q5_flag',
                                'sent_cross_up', 'sent_neg', 'sent_q2_flag_x_close_to_sma10',
                                'sent_cross_up_x_high_low_range', 'sent_neg_x_high_low_range']
            for feature in sentiment_features:
                df[feature] = 0.0
        
        # Target: next day price direction (1 = up, 0 = down)
        df['target'] = (df['close'].shift(-1) > df['close']).astype(int)
        
        # Reset index to make date a column
        df = df.reset_index()
        
        # Select final features (based on BitcoinInsight research)
        self.feature_names = [
            'roc_1d',
            'roc_3d', 
            'high_low_range',
            'bb_width',
            'close_to_sma10_ratio',
            'volume_ratio',
            'sent_5d',
            'sent_q5_flag',
            'sent_cross_up_x_high_low_range',
            'sent_accel',
            'sent_vol',
            'sent_neg_x_high_low_range',
            'sent_q2_flag_x_close_to_sma10'
        ]
        
        # Select features and drop rows with NaN values
        feature_cols = ['date'] + self.feature_names + ['target']
        result_df = df[feature_cols].dropna()
        
        if len(result_df) == 0:
            logger.warning("No valid data after feature preparation")
            
        return result_df
    
    def train_model(self, features_df: pd.DataFrame) -> Dict:
        """
        Train the LightGBM model
        
        Args:
            features_df: DataFrame with features and target
            
        Returns:
            Dict with training metrics
        """
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")
        
        # Prepare data
        X = features_df[self.feature_names].values
        y = features_df['target'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Time series split for validation
        tscv = TimeSeriesSplit(n_splits=3)
        
        # LightGBM parameters (optimized based on research)
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'boosting_type': 'gbdt',
            'num_leaves': 31,
            'learning_rate': 0.1,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42
        }
        
        # Cross-validation
        cv_scores = []
        
        for train_idx, val_idx in tscv.split(X_scaled):
            X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            train_data = lgb.Dataset(X_train, label=y_train)
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            
            model = lgb.train(
                params,
                train_data,
                valid_sets=[val_data],
                num_boost_round=100,
                callbacks=[lgb.early_stopping(10), lgb.log_evaluation(0)]
            )
            
            y_pred = (model.predict(X_val) > 0.5).astype(int)
            accuracy = accuracy_score(y_val, y_pred)
            cv_scores.append(accuracy)
        
        # Train final model on all data
        train_data = lgb.Dataset(X_scaled, label=y)
        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=100,
            callbacks=[lgb.log_evaluation(0)]
        )
        
        self.is_trained = True
        
        # Calculate final metrics
        y_pred_final = (self.model.predict(X_scaled) > 0.5).astype(int)
        
        metrics = {
            'cv_accuracy_mean': np.mean(cv_scores),
            'cv_accuracy_std': np.std(cv_scores),
            'final_accuracy': accuracy_score(y, y_pred_final),
            'final_precision': precision_score(y, y_pred_final),
            'final_recall': recall_score(y, y_pred_final),
            'feature_importance': dict(zip(self.feature_names, 
                                         self.model.feature_importance()))
        }
        
        logger.info(f"Model trained successfully. CV Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
        
        return metrics
    
    def predict_direction(self, features: pd.Series) -> Tuple[int, float]:
        """
        Predict price direction for next day
        
        Args:
            features: Series with feature values
            
        Returns:
            Tuple of (prediction, confidence)
        """
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Prepare features
        feature_values = features[self.feature_names].values.reshape(1, -1)
        feature_values_scaled = self.scaler.transform(feature_values)
        
        # Get prediction probability
        prob_up = self.model.predict(feature_values_scaled)[0]
        prediction = int(prob_up > 0.5)
        confidence = prob_up if prediction == 1 else (1 - prob_up)
        
        return prediction, confidence
    
    def predict_next_direction(self, 
                              price_data: Optional[pd.DataFrame] = None, 
                              sentiment_data: Optional[pd.DataFrame] = None) -> Optional[Dict]:
        """
        Predict Bitcoin price direction for next period
        
        Args:
            price_data: Price data DataFrame
            sentiment_data: Sentiment data DataFrame
            
        Returns:
            Dict with prediction results or None if insufficient data
        """
        try:
            # Validate input data
            if price_data is None or price_data.empty:
                logger.error("Price data required for prediction")
                return None
                
            # Prepare features for prediction
            features_df = self.prepare_features(price_data, sentiment_data)
            
            if features_df.empty:
                logger.warning("Could not prepare features from provided data")
                return None
                
            if len(features_df) < 5:
                logger.warning(f"Insufficient data for reliable prediction: {len(features_df)} samples")
                return None
            
            # Train model if not already trained
            if not self.is_trained:
                logger.info("Training model for first prediction...")
                metrics = self.train_model(features_df)
                if not self.is_trained:
                    logger.error("Model training failed")
                    return None
                logger.info(f"Model trained with accuracy: {metrics.get('cv_accuracy_mean', 'N/A'):.3f}")
            
            # Get latest features for prediction (exclude rows with target NaN)
            valid_features = features_df.dropna(subset=['target'])
            if valid_features.empty:
                # For prediction, we don't need target, use latest available features
                latest_features = features_df.iloc[-1]
            else:
                latest_features = valid_features.iloc[-1]
            
            # Make prediction
            direction, confidence = self.predict_direction(latest_features)
            
            # Get feature importance
            feature_importance = self.get_feature_importance()
            
            result = {
                'direction': direction,
                'direction_text': 'UP' if direction == 1 else 'DOWN',
                'confidence': float(confidence),
                'feature_importance': feature_importance,
                'timestamp': datetime.now().isoformat(),
                'data_points_used': len(features_df)
            }
            
            logger.info(f"Prediction: {result['direction_text']} (confidence: {confidence:.3f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in predict_next_direction: {e}")
            import traceback
            logger.debug(f"Full traceback: {traceback.format_exc()}")
            return None

    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from trained model"""
        if not self.is_trained:
            return {}
        
        importance = self.model.feature_importance()
        return dict(zip(self.feature_names, importance))
    
    def save_model(self, filepath: str):
        """Save trained model"""
        if not self.is_trained:
            raise ValueError("No trained model to save")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'is_trained': self.is_trained
        }
        
        import joblib
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained model"""
        import joblib
        
        model_data = joblib.load(filepath)
        self.model = model_data['model']
        self.scaler = model_data['scaler'] 
        self.feature_names = model_data['feature_names']
        self.is_trained = model_data['is_trained']
        
        logger.info(f"Model loaded from {filepath}")


def create_sample_features() -> pd.DataFrame:
    """Create sample features for testing"""
    dates = pd.date_range(start='2024-01-01', end='2024-12-31', freq='D')
    
    # Sample price data
    np.random.seed(42)
    price_data = pd.DataFrame({
        'date': dates,
        'open': 45000 + np.random.randn(len(dates)) * 1000,
        'high': 46000 + np.random.randn(len(dates)) * 1000,
        'low': 44000 + np.random.randn(len(dates)) * 1000,
        'close': 45000 + np.random.randn(len(dates)) * 1000,
        'volume': 1000000 + np.random.randn(len(dates)) * 100000
    })
    
    # Sample sentiment data
    sentiment_data = pd.DataFrame({
        'date': dates,
        'sentiment': np.random.randn(len(dates)) * 0.5
    })
    
    price_data.set_index('date', inplace=True)
    sentiment_data.set_index('date', inplace=True)
    
    return price_data, sentiment_data


def test_lightgbm_predictor():
    """Test the LightGBM predictor"""
    if not LIGHTGBM_AVAILABLE:
        print("❌ LightGBM not available for testing")
        return
    
    print("🤖 Testing LightGBM Bitcoin Direction Predictor")
    print("=" * 60)
    
    # Create sample data
    price_data, sentiment_data = create_sample_features()
    
    # Initialize predictor
    predictor = LightGBMPredictor()
    
    # Prepare features
    features_df = predictor.prepare_features(price_data, sentiment_data)
    print(f"✅ Features prepared: {len(features_df)} samples, {len(predictor.feature_names)} features")
    
    # Train model
    metrics = predictor.train_model(features_df)
    print(f"✅ Model trained successfully")
    print(f"📊 CV Accuracy: {metrics['cv_accuracy_mean']:.3f} ± {metrics['cv_accuracy_std']:.3f}")
    print(f"📊 Final Accuracy: {metrics['final_accuracy']:.3f}")
    
    # Test prediction
    latest_features = features_df.iloc[-1]
    prediction, confidence = predictor.predict_direction(latest_features)
    direction = "UP ⬆️" if prediction == 1 else "DOWN ⬇️"
    print(f"🔮 Latest prediction: {direction} (confidence: {confidence:.3f})")
    
    # Feature importance
    importance = predictor.get_feature_importance()
    print(f"\n📈 Top 5 Features:")
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)
    for feature, imp in sorted_features[:5]:
        print(f"   {feature}: {imp:.1f}")


if __name__ == "__main__":
    test_lightgbm_predictor()
