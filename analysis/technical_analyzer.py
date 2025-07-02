import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from database.models import get_session, PriceData, TechnicalAnalysis
from config.settings import settings
import ta

class TechnicalAnalyzer:
    def __init__(self):
        self.setup_logging()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def get_price_data(self, days=30):
        """Get price data from database"""
        session = get_session()
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            price_data = session.query(PriceData).filter(
                PriceData.timestamp >= cutoff_time
            ).order_by(PriceData.timestamp).all()
            
            if not price_data:
                self.logger.warning("No price data found in database")
                return None
            
            # Convert to DataFrame
            df = pd.DataFrame([{
                'timestamp': entry.timestamp,
                'open': entry.open_price or entry.price,
                'high': entry.high or entry.price,
                'low': entry.low or entry.price,
                'close': entry.close_price or entry.price,
                'volume': entry.volume or 0
            } for entry in price_data])
            
            # Ensure we have required columns
            df['open'] = df['open'].fillna(df['close'])
            df['high'] = df['high'].fillna(df['close'])
            df['low'] = df['low'].fillna(df['close'])
            
            self.logger.info(f"Retrieved {len(df)} price data points")
            return df
            
        except Exception as e:
            self.logger.error(f"Error retrieving price data: {e}")
            return None
        finally:
            session.close()
    
    def calculate_moving_averages(self, df):
        """Calculate various moving averages"""
        if df is None or len(df) < 50:
            return {}
        
        try:
            ma_data = {}
            
            # Simple Moving Averages
            ma_data['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
            ma_data['sma_50'] = ta.trend.sma_indicator(df['close'], window=50)
            ma_data['sma_200'] = ta.trend.sma_indicator(df['close'], window=200)
            
            # Exponential Moving Averages
            ma_data['ema_12'] = ta.trend.ema_indicator(df['close'], window=12)
            ma_data['ema_26'] = ta.trend.ema_indicator(df['close'], window=26)
            
            # Volume SMA
            if 'volume' in df.columns:
                ma_data['volume_sma'] = ta.trend.sma_indicator(df['volume'], window=20)
            
            return ma_data
            
        except Exception as e:
            self.logger.error(f"Error calculating moving averages: {e}")
            return {}
    
    def calculate_oscillators(self, df):
        """Calculate oscillator indicators"""
        if df is None or len(df) < 30:
            return {}
        
        try:
            osc_data = {}
            
            # RSI
            osc_data['rsi'] = ta.momentum.rsi(df['close'], window=14)
            
            # MACD
            osc_data['macd'] = ta.trend.macd(df['close'])
            osc_data['macd_signal'] = ta.trend.macd_signal(df['close'])
            osc_data['macd_histogram'] = ta.trend.macd_diff(df['close'])
            
            # Stochastic
            osc_data['stoch_k'] = ta.momentum.stoch(df['high'], df['low'], df['close'])
            osc_data['stoch_d'] = ta.momentum.stoch_signal(df['high'], df['low'], df['close'])
            
            # Williams %R
            osc_data['williams_r'] = ta.momentum.williams_r(df['high'], df['low'], df['close'])
            
            # CCI
            osc_data['cci'] = ta.trend.cci(df['high'], df['low'], df['close'])
            
            return osc_data
            
        except Exception as e:
            self.logger.error(f"Error calculating oscillators: {e}")
            return {}
    
    def calculate_volatility_indicators(self, df):
        """Calculate volatility indicators"""
        if df is None or len(df) < 20:
            return {}
        
        try:
            vol_data = {}
            
            # Bollinger Bands
            vol_data['bb_upper'] = ta.volatility.bollinger_hband(df['close'])
            vol_data['bb_middle'] = ta.volatility.bollinger_mavg(df['close'])
            vol_data['bb_lower'] = ta.volatility.bollinger_lband(df['close'])
            
            # ATR
            vol_data['atr'] = ta.volatility.average_true_range(df['high'], df['low'], df['close'])
            
            return vol_data
            
        except Exception as e:
            self.logger.error(f"Error calculating volatility indicators: {e}")
            return {}
    
    def calculate_volume_indicators(self, df):
        """Calculate volume indicators"""
        if df is None or len(df) < 20 or 'volume' not in df.columns:
            return {}
        
        try:
            vol_data = {}
            
            # OBV (On Balance Volume)
            vol_data['obv'] = ta.volume.on_balance_volume(df['close'], df['volume'])
            
            return vol_data
            
        except Exception as e:
            self.logger.error(f"Error calculating volume indicators: {e}")
            return {}
    
    def calculate_trend_indicators(self, df):
        """Calculate trend indicators"""
        if df is None or len(df) < 30:
            return {}
        
        try:
            trend_data = {}
            
            # ADX
            trend_data['adx'] = ta.trend.adx(df['high'], df['low'], df['close'])
            
            return trend_data
            
        except Exception as e:
            self.logger.error(f"Error calculating trend indicators: {e}")
            return {}
    
    def analyze_signals(self, indicators):
        """Analyze technical signals"""
        signals = {
            'bullish_signals': [],
            'bearish_signals': [],
            'neutral_signals': [],
            'strength': 0  # -1 to 1
        }
        
        try:
            # Get latest values (skip NaN values)
            def get_latest_value(series):
                if series is None or len(series) == 0:
                    return None
                for i in range(len(series) - 1, -1, -1):
                    if not pd.isna(series.iloc[i]):
                        return series.iloc[i]
                return None
            
            # RSI Analysis
            latest_rsi = get_latest_value(indicators.get('rsi'))
            if latest_rsi is not None:
                if latest_rsi < 30:
                    signals['bullish_signals'].append(f"RSI oversold ({latest_rsi:.1f})")
                elif latest_rsi > 70:
                    signals['bearish_signals'].append(f"RSI overbought ({latest_rsi:.1f})")
                else:
                    signals['neutral_signals'].append(f"RSI neutral ({latest_rsi:.1f})")
            
            # MACD Analysis
            latest_macd = get_latest_value(indicators.get('macd'))
            latest_macd_signal = get_latest_value(indicators.get('macd_signal'))
            if latest_macd is not None and latest_macd_signal is not None:
                if latest_macd > latest_macd_signal:
                    signals['bullish_signals'].append("MACD bullish crossover")
                else:
                    signals['bearish_signals'].append("MACD bearish crossover")
            
            # Bollinger Bands Analysis
            latest_close = get_latest_value(indicators.get('close'))
            latest_bb_upper = get_latest_value(indicators.get('bb_upper'))
            latest_bb_lower = get_latest_value(indicators.get('bb_lower'))
            
            if all(v is not None for v in [latest_close, latest_bb_upper, latest_bb_lower]):
                if latest_close > latest_bb_upper:
                    signals['bearish_signals'].append("Price above upper Bollinger Band")
                elif latest_close < latest_bb_lower:
                    signals['bullish_signals'].append("Price below lower Bollinger Band")
                else:
                    signals['neutral_signals'].append("Price within Bollinger Bands")
            
            # Moving Average Analysis
            latest_sma_20 = get_latest_value(indicators.get('sma_20'))
            latest_sma_50 = get_latest_value(indicators.get('sma_50'))
            
            if latest_sma_20 is not None and latest_sma_50 is not None:
                if latest_sma_20 > latest_sma_50:
                    signals['bullish_signals'].append("20 SMA above 50 SMA (Golden Cross)")
                else:
                    signals['bearish_signals'].append("20 SMA below 50 SMA (Death Cross)")
            
            # Calculate overall strength
            bullish_count = len(signals['bullish_signals'])
            bearish_count = len(signals['bearish_signals'])
            total_signals = bullish_count + bearish_count
            
            if total_signals > 0:
                signals['strength'] = (bullish_count - bearish_count) / total_signals
            
            return signals
            
        except Exception as e:
            self.logger.error(f"Error analyzing signals: {e}")
            return signals
    
    def save_to_database(self, indicators, timeframe='1d'):
        """Save technical analysis to database"""
        session = get_session()
        
        try:
            # Get latest values
            def get_latest_value(series):
                if series is None or len(series) == 0:
                    return None
                for i in range(len(series) - 1, -1, -1):
                    if not pd.isna(series.iloc[i]):
                        return float(series.iloc[i])
                return None
            
            ta_entry = TechnicalAnalysis(
                symbol=settings.SYMBOL,
                timeframe=timeframe,
                sma_20=get_latest_value(indicators.get('sma_20')),
                sma_50=get_latest_value(indicators.get('sma_50')),
                sma_200=get_latest_value(indicators.get('sma_200')),
                ema_12=get_latest_value(indicators.get('ema_12')),
                ema_26=get_latest_value(indicators.get('ema_26')),
                rsi=get_latest_value(indicators.get('rsi')),
                macd=get_latest_value(indicators.get('macd')),
                macd_signal=get_latest_value(indicators.get('macd_signal')),
                macd_histogram=get_latest_value(indicators.get('macd_histogram')),
                stoch_k=get_latest_value(indicators.get('stoch_k')),
                stoch_d=get_latest_value(indicators.get('stoch_d')),
                williams_r=get_latest_value(indicators.get('williams_r')),
                bb_upper=get_latest_value(indicators.get('bb_upper')),
                bb_middle=get_latest_value(indicators.get('bb_middle')),
                bb_lower=get_latest_value(indicators.get('bb_lower')),
                atr=get_latest_value(indicators.get('atr')),
                obv=get_latest_value(indicators.get('obv')),
                volume_sma=get_latest_value(indicators.get('volume_sma')),
                adx=get_latest_value(indicators.get('adx')),
                cci=get_latest_value(indicators.get('cci'))
            )
            
            session.add(ta_entry)
            session.commit()
            
            self.logger.info("Technical analysis saved to database")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving technical analysis: {e}")
        finally:
            session.close()
    
    def perform_full_analysis(self, days=30):
        """Perform complete technical analysis"""
        self.logger.info("Starting technical analysis...")
        
        # Get price data
        df = self.get_price_data(days)
        if df is None:
            return None
        
        # Calculate all indicators
        indicators = {}
        indicators.update(self.calculate_moving_averages(df))
        indicators.update(self.calculate_oscillators(df))
        indicators.update(self.calculate_volatility_indicators(df))
        indicators.update(self.calculate_volume_indicators(df))
        indicators.update(self.calculate_trend_indicators(df))
        
        # Add price data for signal analysis
        indicators['close'] = df['close']
        
        # Analyze signals
        signals = self.analyze_signals(indicators)
        
        # Save to database
        self.save_to_database(indicators)
        
        # Generate summary
        summary = {
            'timestamp': datetime.utcnow(),
            'data_points': len(df),
            'indicators': {k: v.iloc[-1] if hasattr(v, 'iloc') and len(v) > 0 else v 
                          for k, v in indicators.items() if k != 'close'},
            'signals': signals,
            'recommendation': self._get_recommendation(signals)
        }
        
        self.logger.info(f"Technical analysis complete: {signals['strength']:.2f} strength")
        return summary
    
    def _get_recommendation(self, signals):
        """Generate trading recommendation based on signals"""
        strength = signals['strength']
        
        if strength > 0.3:
            return "BUY"
        elif strength < -0.3:
            return "SELL"
        else:
            return "HOLD"

if __name__ == "__main__":
    analyzer = TechnicalAnalyzer()
    analysis = analyzer.perform_full_analysis(days=30)
    if analysis:
        print(f"Technical Analysis Result: {analysis['recommendation']}")
        print(f"Signal Strength: {analysis['signals']['strength']:.2f}")
        print(f"Bullish Signals: {len(analysis['signals']['bullish_signals'])}")
        print(f"Bearish Signals: {len(analysis['signals']['bearish_signals'])}")
