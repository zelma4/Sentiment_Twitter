"""
Advanced Feature Engineering for Crypto Prediction
Includes Google Trends, Wikipedia, On-chain metrics, and advanced indicators
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import warnings
warnings.filterwarnings('ignore')

# Google Trends
try:
    from pytrends.request import TrendReq
    GOOGLE_TRENDS_AVAILABLE = True
except ImportError:
    GOOGLE_TRENDS_AVAILABLE = False

# Wikipedia
try:
    import wikipedia
    import requests
    WIKIPEDIA_AVAILABLE = True
except ImportError:
    WIKIPEDIA_AVAILABLE = False

# Technical analysis
try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False

# Advanced sentiment
try:
    from analysis.advanced_sentiment import AdvancedSentimentAnalyzer
    ADVANCED_SENTIMENT_AVAILABLE = True
except ImportError:
    ADVANCED_SENTIMENT_AVAILABLE = False

class AdvancedFeatureEngineer:
    """
    Advanced feature engineering for crypto prediction
    Based on research showing importance of multi-modal features
    """
    
    def __init__(self):
        self.setup_logging()
        
        # Initialize components
        self.trends_client = None
        self.sentiment_analyzer = None
        
        self._initialize_components()
        
        # Cache for external data
        self.cache = {}
        self.cache_timeout = 3600  # 1 hour
        
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
    
    def _initialize_components(self):
        """Initialize all components"""
        # Google Trends
        if GOOGLE_TRENDS_AVAILABLE:
            try:
                self.trends_client = TrendReq(hl='en-US', tz=360)
                self.logger.info("✅ Google Trends initialized")
            except Exception as e:
                self.logger.error(f"❌ Google Trends initialization failed: {e}")
                self.trends_client = None
        
        # Advanced sentiment analyzer
        if ADVANCED_SENTIMENT_AVAILABLE:
            try:
                self.sentiment_analyzer = AdvancedSentimentAnalyzer()
                self.logger.info("✅ Advanced sentiment analyzer initialized")
            except Exception as e:
                self.logger.error(f"❌ Advanced sentiment analyzer failed: {e}")
                self.sentiment_analyzer = None
    
    def engineer_all_features(self, 
                            price_data: pd.DataFrame,
                            sentiment_data: Optional[pd.DataFrame] = None,
                            crypto_symbol: str = 'bitcoin') -> Dict:
        """
        Engineer all available features for crypto prediction
        
        Args:
            price_data: Price data with OHLCV
            sentiment_data: Sentiment data
            crypto_symbol: Crypto symbol for external data
            
        Returns:
            Dictionary with all engineered features
        """
        try:
            self.logger.info(f"🔧 Engineering features for {crypto_symbol}")
            
            features = {}
            
            # 1. Technical indicators (Hard features)
            tech_features = self._engineer_technical_features(price_data)
            features.update(tech_features)
            
            # 2. Price momentum and volatility features
            momentum_features = self._engineer_momentum_features(price_data)
            features.update(momentum_features)
            
            # 3. Volume features
            volume_features = self._engineer_volume_features(price_data)
            features.update(volume_features)
            
            # 4. Sentiment features (if available)
            if sentiment_data is not None:
                sentiment_features = self._engineer_sentiment_features(sentiment_data)
                features.update(sentiment_features)
            
            # 5. Google Trends features
            trends_features = self._engineer_trends_features(crypto_symbol)
            features.update(trends_features)
            
            # 6. Wikipedia features
            wiki_features = self._engineer_wikipedia_features(crypto_symbol)
            features.update(wiki_features)
            
            # 7. On-chain proxy features (using price data)
            onchain_features = self._engineer_onchain_proxy_features(price_data)
            features.update(onchain_features)
            
            # 8. Market structure features
            structure_features = self._engineer_market_structure_features(price_data)
            features.update(structure_features)
            
            # 9. Time-based features
            time_features = self._engineer_time_features()
            features.update(time_features)
            
            # 10. Composite features (combinations)
            composite_features = self._engineer_composite_features(features)
            features.update(composite_features)
            
            self.logger.info(f"✅ Engineered {len(features)} features")
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature engineering failed: {e}")
            return self._get_default_features()
    
    def _engineer_technical_features(self, price_data: pd.DataFrame) -> Dict:
        """Engineer technical analysis features"""
        try:
            features = {}
            
            # Ensure we have required columns
            if 'close' not in price_data.columns:
                return features
            
            close_prices = price_data['close'].values
            high_prices = price_data['high'].values if 'high' in price_data.columns else close_prices
            low_prices = price_data['low'].values if 'low' in price_data.columns else close_prices
            volume = price_data['volume'].values if 'volume' in price_data.columns else np.ones(len(close_prices))
            
            # Moving averages
            for period in [5, 10, 20, 50, 100, 200]:
                if len(close_prices) >= period:
                    sma = pd.Series(close_prices).rolling(period).mean().iloc[-1]
                    features[f'sma_{period}'] = sma
                    features[f'price_to_sma_{period}'] = close_prices[-1] / sma if sma > 0 else 1.0
            
            # RSI
            if len(close_prices) >= 14:
                rsi = self._calculate_rsi(close_prices, 14)
                features['rsi'] = rsi
                features['rsi_overbought'] = 1.0 if rsi > 70 else 0.0
                features['rsi_oversold'] = 1.0 if rsi < 30 else 0.0
            
            # MACD
            if len(close_prices) >= 26:
                macd, macd_signal, macd_hist = self._calculate_macd(close_prices)
                features['macd'] = macd
                features['macd_signal'] = macd_signal
                features['macd_histogram'] = macd_hist
                features['macd_bullish'] = 1.0 if macd > macd_signal else 0.0
            
            # Bollinger Bands
            if len(close_prices) >= 20:
                bb_upper, bb_middle, bb_lower = self._calculate_bollinger_bands(close_prices)
                features['bb_upper'] = bb_upper
                features['bb_middle'] = bb_middle
                features['bb_lower'] = bb_lower
                features['bb_position'] = (close_prices[-1] - bb_lower) / (bb_upper - bb_lower) if bb_upper > bb_lower else 0.5
            
            return features
            
        except Exception as e:
            self.logger.error(f"Technical features error: {e}")
            return {}
    
    def _engineer_momentum_features(self, price_data: pd.DataFrame) -> Dict:
        """Engineer momentum and volatility features"""
        try:
            features = {}
            
            if 'close' not in price_data.columns:
                return features
            
            close_prices = pd.Series(price_data['close'].values)
            
            # Price returns for different periods
            for period in [1, 4, 24, 168]:  # 1h, 4h, 1d, 1w
                if len(close_prices) > period:
                    returns = close_prices.pct_change(period).iloc[-1]
                    features[f'returns_{period}h'] = returns
                    
                    # Rolling volatility
                    volatility = close_prices.pct_change().rolling(period).std().iloc[-1]
                    features[f'volatility_{period}h'] = volatility
            
            # Volatility clustering (important for crypto)
            if len(close_prices) >= 24:
                returns = close_prices.pct_change().dropna()
                volatility_clustering = self._calculate_volatility_clustering(returns)
                features['volatility_clustering'] = volatility_clustering
            
            return features
            
        except Exception as e:
            self.logger.error(f"Momentum features error: {e}")
            return {}
    
    def _engineer_volume_features(self, price_data: pd.DataFrame) -> Dict:
        """Engineer volume-based features"""
        try:
            features = {}
            
            if 'volume' not in price_data.columns:
                return features
            
            volume = pd.Series(price_data['volume'].values)
            
            # Volume moving averages
            for period in [5, 20, 50]:
                if len(volume) >= period:
                    vol_ma = volume.rolling(period).mean().iloc[-1]
                    features[f'volume_ma_{period}'] = vol_ma
                    features[f'volume_ratio_{period}'] = volume.iloc[-1] / vol_ma if vol_ma > 0 else 1.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Volume features error: {e}")
            return {}
    
    def _engineer_sentiment_features(self, sentiment_data: pd.DataFrame) -> Dict:
        """Engineer sentiment-based features"""
        try:
            features = {}
            
            if sentiment_data is None or sentiment_data.empty:
                return features
            
            # Basic sentiment features
            if 'sentiment' in sentiment_data.columns:
                sentiment_series = sentiment_data['sentiment']
                
                # Current sentiment
                features['sentiment_current'] = sentiment_series.iloc[-1]
                
                # Sentiment momentum (key feature from research)
                if len(sentiment_series) >= 2:
                    sentiment_momentum = sentiment_series.diff().iloc[-1]
                    features['sentiment_momentum'] = sentiment_momentum
                
                # Sentiment moving averages
                for period in [5, 24, 168]:
                    if len(sentiment_series) >= period:
                        sentiment_ma = sentiment_series.rolling(period).mean().iloc[-1]
                        features[f'sentiment_ma_{period}'] = sentiment_ma
            
            return features
            
        except Exception as e:
            self.logger.error(f"Sentiment features error: {e}")
            return {}
    
    def _engineer_trends_features(self, crypto_symbol: str) -> Dict:
        """Engineer Google Trends features"""
        try:
            features = {}
            
            if not GOOGLE_TRENDS_AVAILABLE or not self.trends_client:
                return features
            
            # Mock implementation for now
            features['google_trends'] = 50.0  # Default value
            
            return features
            
        except Exception as e:
            self.logger.error(f"Trends features error: {e}")
            return {}
    
    def _engineer_wikipedia_features(self, crypto_symbol: str) -> Dict:
        """Engineer Wikipedia features"""
        try:
            features = {}
            
            if not WIKIPEDIA_AVAILABLE:
                return features
            
            # Mock implementation for now
            features['wikipedia_views'] = 1000.0  # Default value
            
            return features
            
        except Exception as e:
            self.logger.error(f"Wikipedia features error: {e}")
            return {}
    
    def _engineer_onchain_proxy_features(self, price_data: pd.DataFrame) -> Dict:
        """Engineer on-chain proxy features using price data"""
        try:
            features = {}
            
            if 'close' not in price_data.columns:
                return features
            
            close_prices = pd.Series(price_data['close'].values)
            volume = pd.Series(price_data['volume'].values) if 'volume' in price_data.columns else pd.Series(np.ones(len(close_prices)))
            
            # NVT Proxy (using volume as proxy for transaction volume)
            if len(close_prices) >= 30:
                market_cap_proxy = close_prices.iloc[-1]  # Using price as proxy
                volume_ma_30 = volume.rolling(30).mean().iloc[-1]
                nvt_proxy = market_cap_proxy / volume_ma_30 if volume_ma_30 > 0 else 1.0
                features['nvt_proxy'] = nvt_proxy
            
            # MVRV Proxy (using price moving averages)
            if len(close_prices) >= 365:
                price_current = close_prices.iloc[-1]
                price_ma_365 = close_prices.rolling(365).mean().iloc[-1]
                mvrv_proxy = price_current / price_ma_365 if price_ma_365 > 0 else 1.0
                features['mvrv_proxy'] = mvrv_proxy
            
            return features
            
        except Exception as e:
            self.logger.error(f"On-chain proxy features error: {e}")
            return {}
    
    def _engineer_market_structure_features(self, price_data: pd.DataFrame) -> Dict:
        """Engineer market structure features"""
        try:
            features = {}
            
            if 'close' not in price_data.columns:
                return features
            
            close_prices = pd.Series(price_data['close'].values)
            
            # Support and Resistance levels
            if len(close_prices) >= 50:
                support_level = close_prices.rolling(50).min().iloc[-1]
                resistance_level = close_prices.rolling(50).max().iloc[-1]
                features['support_level'] = support_level
                features['resistance_level'] = resistance_level
            
            return features
            
        except Exception as e:
            self.logger.error(f"Market structure features error: {e}")
            return {}
    
    def _engineer_time_features(self) -> Dict:
        """Engineer time-based features"""
        try:
            features = {}
            
            now = datetime.utcnow()
            
            # Time of day features
            features['hour'] = now.hour
            features['day_of_week'] = now.weekday()
            features['weekend'] = 1.0 if now.weekday() >= 5 else 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Time features error: {e}")
            return {}
    
    def _engineer_composite_features(self, features: Dict) -> Dict:
        """Engineer composite features from existing features"""
        try:
            composite = {}
            
            # Technical strength score
            technical_score = 0
            if 'rsi' in features:
                technical_score += 1 if 30 <= features['rsi'] <= 70 else 0
            if 'macd_bullish' in features:
                technical_score += features['macd_bullish']
            
            composite['technical_strength'] = technical_score
            
            # Momentum score
            momentum_score = 0
            for key in features:
                if 'momentum' in key.lower() and isinstance(features[key], (int, float)):
                    momentum_score += 1 if features[key] > 0 else -1
            
            composite['momentum_score'] = momentum_score
            
            return composite
            
        except Exception as e:
            self.logger.error(f"Composite features error: {e}")
            return {}
    
    # Helper methods for calculations
    def _calculate_rsi(self, prices: np.ndarray, period: int = 14) -> float:
        """Calculate RSI"""
        try:
            delta = np.diff(prices)
            gain = np.where(delta > 0, delta, 0)
            loss = np.where(delta < 0, -delta, 0)
            
            avg_gain = np.mean(gain[-period:])
            avg_loss = np.mean(loss[-period:])
            
            if avg_loss == 0:
                return 100.0
            
            rs = avg_gain / avg_loss
            rsi = 100 - (100 / (1 + rs))
            
            return float(rsi)
        except:
            return 50.0
    
    def _calculate_macd(self, prices: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """Calculate MACD"""
        try:
            prices_series = pd.Series(prices)
            ema_fast = prices_series.ewm(span=fast).mean()
            ema_slow = prices_series.ewm(span=slow).mean()
            
            macd = ema_fast - ema_slow
            macd_signal = macd.ewm(span=signal).mean()
            macd_hist = macd - macd_signal
            
            return float(macd.iloc[-1]), float(macd_signal.iloc[-1]), float(macd_hist.iloc[-1])
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_bollinger_bands(self, prices: np.ndarray, period: int = 20, std_dev: float = 2.0) -> tuple:
        """Calculate Bollinger Bands"""
        try:
            prices_series = pd.Series(prices)
            sma = prices_series.rolling(period).mean()
            std = prices_series.rolling(period).std()
            
            upper = sma + (std * std_dev)
            lower = sma - (std * std_dev)
            
            return float(upper.iloc[-1]), float(sma.iloc[-1]), float(lower.iloc[-1])
        except:
            return 0.0, 0.0, 0.0
    
    def _calculate_volatility_clustering(self, returns: pd.Series) -> float:
        """Calculate volatility clustering measure"""
        try:
            # Calculate absolute returns
            abs_returns = abs(returns)
            
            # Calculate autocorrelation of squared returns
            autocorr = abs_returns.autocorr(lag=1)
            
            return float(autocorr) if not np.isnan(autocorr) else 0.0
        except:
            return 0.0
    
    def _get_default_features(self) -> Dict:
        """Return default features when engineering fails"""
        return {
            'price': 0.0,
            'volume': 0.0,
            'rsi': 50.0,
            'macd': 0.0,
            'bb_position': 0.5,
            'sentiment_momentum': 0.0,
            'volatility_clustering': 0.0,
            'technical_strength': 0,
            'momentum_score': 0,
            'volatility_score': 0.0
        }
