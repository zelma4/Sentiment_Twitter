"""
Enhanced Data Collector for Cryptocurrency Analysis

This module collects additional data sources for improved analysis:
- Fear & Greed Index
- On-chain metrics
- StockTwits sentiment
- Market correlation data
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time

logger = logging.getLogger(__name__)


class FearGreedCollector:
    """Collector for Crypto Fear & Greed Index"""
    
    def __init__(self):
        self.base_url = "https://api.alternative.me/fng/"
        
    def get_current_fear_greed(self) -> Dict:
        """Get current Fear & Greed index"""
        try:
            response = requests.get(f"{self.base_url}?limit=1", timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data['data']:
                fg_data = data['data'][0]
                return {
                    'value': int(fg_data['value']),
                    'classification': fg_data['value_classification'],
                    'timestamp': datetime.fromtimestamp(int(fg_data['timestamp'])),
                    'score_normalized': (int(fg_data['value']) - 50) / 50  # -1 to 1
                }
        except Exception as e:
            logger.error(f"Failed to fetch Fear & Greed index: {e}")
            
        return {
            'value': 50,
            'classification': 'Neutral',
            'timestamp': datetime.now(),
            'score_normalized': 0.0
        }
    
    def get_historical_fear_greed(self, days: int = 30) -> pd.DataFrame:
        """Get historical Fear & Greed data"""
        try:
            response = requests.get(f"{self.base_url}?limit={days}", timeout=15)
            response.raise_for_status()
            
            data = response.json()
            
            df_data = []
            for item in data['data']:
                df_data.append({
                    'date': datetime.fromtimestamp(int(item['timestamp'])).date(),
                    'fear_greed_value': int(item['value']),
                    'fear_greed_class': item['value_classification'],
                    'fear_greed_normalized': (int(item['value']) - 50) / 50
                })
            
            df = pd.DataFrame(df_data)
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            logger.info(f"✅ Collected {len(df)} days of Fear & Greed data")
            return df
            
        except Exception as e:
            logger.error(f"Failed to fetch historical Fear & Greed: {e}")
            return pd.DataFrame()


class OnChainCollector:
    """Collector for Bitcoin on-chain metrics"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.base_url = "https://api.glassnode.com/v1/metrics"
        
    def get_network_value_to_transactions(self) -> Dict:
        """Get NVT ratio (Network Value to Transactions)"""
        if not self.api_key:
            logger.warning("No Glassnode API key provided")
            return {'nvt_ratio': 50.0, 'timestamp': datetime.now()}
            
        try:
            url = f"{self.base_url}/indicators/nvt"
            params = {
                'a': 'BTC',
                'api_key': self.api_key,
                'f': 'JSON',
                's': int((datetime.now() - timedelta(days=1)).timestamp())
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                latest = data[-1]
                return {
                    'nvt_ratio': latest['v'],
                    'timestamp': datetime.fromtimestamp(latest['t'])
                }
                
        except Exception as e:
            logger.error(f"Failed to fetch NVT ratio: {e}")
            
        return {'nvt_ratio': 50.0, 'timestamp': datetime.now()}
    
    def get_mvrv_ratio(self) -> Dict:
        """Get MVRV ratio (Market Value to Realized Value)"""
        if not self.api_key:
            return {'mvrv_ratio': 1.0, 'timestamp': datetime.now()}
            
        try:
            url = f"{self.base_url}/market/mvrv"
            params = {
                'a': 'BTC',
                'api_key': self.api_key,
                'f': 'JSON',
                's': int((datetime.now() - timedelta(days=1)).timestamp())
            }
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            if data:
                latest = data[-1]
                return {
                    'mvrv_ratio': latest['v'],
                    'timestamp': datetime.fromtimestamp(latest['t'])
                }
                
        except Exception as e:
            logger.error(f"Failed to fetch MVRV ratio: {e}")
            
        return {'mvrv_ratio': 1.0, 'timestamp': datetime.now()}


class StockTwitsCollector:
    """Collector for StockTwits cryptocurrency sentiment"""
    
    def __init__(self):
        self.base_url = "https://api.stocktwits.com/api/2"
        self.symbols = ['BTC.X', 'ETH.X', 'CRYPTO']
        
    def get_stream_data(self, symbol: str = 'BTC.X', limit: int = 30) -> List[Dict]:
        """Get recent messages from StockTwits"""
        try:
            url = f"{self.base_url}/streams/symbol/{symbol}.json"
            params = {'limit': limit}
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                messages = []
                
                for msg in data.get('messages', []):
                    sentiment = None
                    if msg.get('entities') and msg['entities'].get('sentiment'):
                        sentiment = msg['entities']['sentiment']['basic']
                    
                    messages.append({
                        'id': msg['id'],
                        'body': msg['body'],
                        'sentiment': sentiment,
                        'created_at': msg['created_at'],
                        'user_followers': msg['user'].get('followers', 0),
                        'symbol': symbol
                    })
                
                logger.info(f"✅ Collected {len(messages)} StockTwits messages for {symbol}")
                return messages
                
        except Exception as e:
            logger.error(f"Failed to fetch StockTwits data for {symbol}: {e}")
            
        return []
    
    def get_aggregated_sentiment(self) -> Dict:
        """Get aggregated sentiment from multiple crypto symbols"""
        all_messages = []
        
        for symbol in self.symbols[:2]:  # Limit to prevent rate limiting
            messages = self.get_stream_data(symbol)
            all_messages.extend(messages)
            time.sleep(1)  # Rate limiting
        
        if not all_messages:
            return {
                'bullish_ratio': 0.5,
                'bearish_ratio': 0.5,
                'total_messages': 0,
                'avg_followers': 0,
                'sentiment_score': 0.0
            }
        
        # Calculate sentiment ratios
        sentiment_counts = {'Bullish': 0, 'Bearish': 0, 'None': 0}
        total_followers = 0
        
        for msg in all_messages:
            sentiment = msg.get('sentiment', 'None')
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
            total_followers += msg.get('user_followers', 0)
        
        total_with_sentiment = sentiment_counts['Bullish'] + sentiment_counts['Bearish']
        
        if total_with_sentiment > 0:
            bullish_ratio = sentiment_counts['Bullish'] / total_with_sentiment
            bearish_ratio = sentiment_counts['Bearish'] / total_with_sentiment
        else:
            bullish_ratio = bearish_ratio = 0.5
        
        return {
            'bullish_ratio': bullish_ratio,
            'bearish_ratio': bearish_ratio,
            'total_messages': len(all_messages),
            'avg_followers': total_followers / len(all_messages) if all_messages else 0,
            'sentiment_score': bullish_ratio - bearish_ratio  # -1 to 1
        }


class MarketCorrelationCollector:
    """Collector for market correlation data"""
    
    def __init__(self):
        self.symbols = ['SPY', 'QQQ', 'GLD', 'DXY']  # S&P 500, Nasdaq, Gold, Dollar
        
    def get_market_data(self, symbol: str, period: str = '5d') -> pd.DataFrame:
        """Get market data for correlation analysis"""
        try:
            import yfinance as yf
            
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            
            if not hist.empty:
                return hist[['Close']].rename(columns={'Close': symbol})
                
        except Exception as e:
            logger.error(f"Failed to fetch market data for {symbol}: {e}")
            
        return pd.DataFrame()
    
    def calculate_btc_correlations(self, btc_prices: pd.DataFrame) -> Dict:
        """Calculate Bitcoin correlations with traditional markets"""
        correlations = {}
        
        # Ensure btc_prices index is timezone-naive
        try:
            if hasattr(btc_prices.index, 'tz') and btc_prices.index.tz is not None:
                btc_prices = btc_prices.copy()
                btc_prices.index = btc_prices.index.tz_localize(None)
        except Exception:
            pass
        
        for symbol in self.symbols:
            try:
                market_data = self.get_market_data(symbol)
                
                if not market_data.empty and not btc_prices.empty:
                    # Ensure market_data index is timezone-naive
                    try:
                        if hasattr(market_data.index, 'tz') and market_data.index.tz is not None:
                            market_data.index = market_data.index.tz_localize(None)
                    except Exception:
                        pass
                    
                    # Merge on index (dates)
                    merged = btc_prices.merge(
                        market_data, 
                        left_index=True, 
                        right_index=True, 
                        how='inner'
                    )
                    
                    if len(merged) > 1:
                        correlation = merged.iloc[:, 0].corr(merged.iloc[:, 1])
                        key = f'btc_{symbol.lower()}_corr'
                        correlations[key] = correlation
                        
            except Exception as e:
                error_msg = f"Failed to calculate correlation with {symbol}: {e}"
                logger.error(error_msg)
                correlations[f'btc_{symbol.lower()}_corr'] = 0.0
        
        logger.info(f"✅ Calculated correlations: {list(correlations.keys())}")
        return correlations


class EnhancedDataCollector:
    """Main collector that orchestrates all data sources"""
    
    def __init__(self, glassnode_api_key: Optional[str] = None):
        self.fear_greed = FearGreedCollector()
        self.onchain = OnChainCollector(glassnode_api_key)
        self.stocktwits = StockTwitsCollector()
        self.market_corr = MarketCorrelationCollector()
        
    def collect_all_metrics(self, btc_prices: Optional[pd.DataFrame] = None) -> Dict:
        """Collect all enhanced metrics"""
        logger.info("🔄 Collecting enhanced market metrics...")
        
        metrics = {}
        
        # Fear & Greed Index
        try:
            fg_data = self.fear_greed.get_current_fear_greed()
            metrics.update({
                'fear_greed_value': fg_data['value'],
                'fear_greed_normalized': fg_data['score_normalized'],
                'fear_greed_class': fg_data['classification']
            })
        except Exception as e:
            logger.error(f"Fear & Greed collection failed: {e}")
            metrics.update({
                'fear_greed_value': 50,
                'fear_greed_normalized': 0.0,
                'fear_greed_class': 'Neutral'
            })
        
        # On-chain metrics
        try:
            nvt_data = self.onchain.get_network_value_to_transactions()
            mvrv_data = self.onchain.get_mvrv_ratio()
            
            metrics.update({
                'nvt_ratio': nvt_data['nvt_ratio'],
                'mvrv_ratio': mvrv_data['mvrv_ratio']
            })
        except Exception as e:
            logger.error(f"On-chain collection failed: {e}")
            metrics.update({
                'nvt_ratio': 50.0,
                'mvrv_ratio': 1.0
            })
        
        # StockTwits sentiment
        try:
            st_sentiment = self.stocktwits.get_aggregated_sentiment()
            metrics.update({
                'stocktwits_sentiment': st_sentiment['sentiment_score'],
                'stocktwits_bullish_ratio': st_sentiment['bullish_ratio'],
                'stocktwits_messages_count': st_sentiment['total_messages']
            })
        except Exception as e:
            logger.error(f"StockTwits collection failed: {e}")
            metrics.update({
                'stocktwits_sentiment': 0.0,
                'stocktwits_bullish_ratio': 0.5,
                'stocktwits_messages_count': 0
            })
        
        # Market correlations
        if btc_prices is not None:
            try:
                correlations = self.market_corr.calculate_btc_correlations(btc_prices)
                metrics.update(correlations)
            except Exception as e:
                logger.error(f"Market correlation calculation failed: {e}")
                # Add default correlations
                for symbol in ['spy', 'qqq', 'gld', 'dxy']:
                    metrics[f'btc_{symbol}_corr'] = 0.0
        
        logger.info(f"✅ Enhanced metrics collected: {len(metrics)} indicators")
        return metrics


def test_enhanced_collectors():
    """Test all enhanced data collectors"""
    print("🧪 Testing Enhanced Data Collectors")
    print("=" * 60)
    
    # Test Fear & Greed
    fg_collector = FearGreedCollector()
    fg_data = fg_collector.get_current_fear_greed()
    print(f"Fear & Greed: {fg_data['value']} ({fg_data['classification']})")
    
    # Test StockTwits
    st_collector = StockTwitsCollector()
    st_data = st_collector.get_aggregated_sentiment()
    print(f"StockTwits Sentiment: {st_data['sentiment_score']:.3f}")
    print(f"Messages analyzed: {st_data['total_messages']}")
    
    # Test complete collection
    collector = EnhancedDataCollector()
    all_metrics = collector.collect_all_metrics()
    
    print(f"\n📊 All Enhanced Metrics ({len(all_metrics)}):")
    for key, value in all_metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")


if __name__ == "__main__":
    test_enhanced_collectors()
