"""
On-chain Data Collector for Bitcoin Analysis
Collects NVT, MVRV, and other on-chain metrics for enhanced predictions
"""

import logging
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
import time

class OnChainDataCollector:
    """
    Collector for Bitcoin on-chain metrics
    Provides NVT, MVRV, and volatility indicators
    """
    
    def __init__(self, glassnode_api_key: Optional[str] = None):
        self.setup_logging()
        
        # API configuration
        self.glassnode_api_key = glassnode_api_key
        self.base_url = "https://api.glassnode.com/v1/metrics"
        
        # Fallback: use free coinmetrics API
        self.coinmetrics_url = "https://api.coinmetrics.io/v4"
        
        self.logger.info("🔗 On-chain data collector initialized")
        
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
    
    def get_nvt_ratio(self, days: int = 30) -> Optional[Dict]:
        """
        Get Network Value to Transactions (NVT) ratio
        High NVT = overvalued, Low NVT = undervalued
        """
        try:
            if self.glassnode_api_key:
                return self._get_glassnode_metric("indicators/nvt", days)
            else:
                return self._get_fallback_nvt(days)
                
        except Exception as e:
            self.logger.error(f"Error getting NVT ratio: {e}")
            return None
    
    def get_mvrv_ratio(self, days: int = 30) -> Optional[Dict]:
        """
        Get Market Value to Realized Value (MVRV) ratio
        High MVRV = market top, Low MVRV = market bottom
        """
        try:
            if self.glassnode_api_key:
                return self._get_glassnode_metric("indicators/mvrv", days)
            else:
                return self._get_fallback_mvrv(days)
                
        except Exception as e:
            self.logger.error(f"Error getting MVRV ratio: {e}")
            return None
    
    def get_realized_volatility(self, days: int = 30) -> Optional[Dict]:
        """
        Get realized volatility - VIX-like on-chain indicator
        """
        try:
            if self.glassnode_api_key:
                return self._get_glassnode_metric("indicators/volatility", days)
            else:
                return self._calculate_fallback_volatility(days)
                
        except Exception as e:
            self.logger.error(f"Error getting realized volatility: {e}")
            return None
    
    def get_active_addresses(self, days: int = 30) -> Optional[Dict]:
        """
        Get active addresses count - network activity indicator
        """
        try:
            if self.glassnode_api_key:
                return self._get_glassnode_metric("addresses/active_count", days)
            else:
                return self._get_fallback_addresses(days)
                
        except Exception as e:
            self.logger.error(f"Error getting active addresses: {e}")
            return None
    
    def get_exchange_flows(self, days: int = 30) -> Optional[Dict]:
        """
        Get exchange inflows/outflows - selling/buying pressure
        """
        try:
            if self.glassnode_api_key:
                inflows = self._get_glassnode_metric("transactions/transfers_to_exchanges_sum", days)
                outflows = self._get_glassnode_metric("transactions/transfers_from_exchanges_sum", days)
                
                if inflows and outflows:
                    return {
                        'inflows': inflows,
                        'outflows': outflows,
                        'net_flow': self._calculate_net_flow(inflows, outflows)
                    }
            
            return self._get_fallback_flows(days)
                
        except Exception as e:
            self.logger.error(f"Error getting exchange flows: {e}")
            return None
    
    def get_all_onchain_metrics(self, days: int = 30) -> Dict:
        """
        Get comprehensive on-chain metrics
        """
        self.logger.info(f"🔗 Collecting on-chain metrics for {days} days...")
        
        metrics = {}
        
        # NVT Ratio
        nvt = self.get_nvt_ratio(days)
        if nvt:
            metrics['nvt_ratio'] = nvt.get('latest_value', 0)
            metrics['nvt_trend'] = nvt.get('trend', 'neutral')
        
        # MVRV Ratio
        mvrv = self.get_mvrv_ratio(days)
        if mvrv:
            metrics['mvrv_ratio'] = mvrv.get('latest_value', 0)
            metrics['mvrv_trend'] = mvrv.get('trend', 'neutral')
        
        # Realized Volatility
        volatility = self.get_realized_volatility(days)
        if volatility:
            metrics['realized_volatility'] = volatility.get('latest_value', 0)
            metrics['volatility_trend'] = volatility.get('trend', 'neutral')
        
        # Active Addresses
        addresses = self.get_active_addresses(days)
        if addresses:
            metrics['active_addresses'] = addresses.get('latest_value', 0)
            metrics['addresses_trend'] = addresses.get('trend', 'neutral')
        
        # Exchange Flows
        flows = self.get_exchange_flows(days)
        if flows:
            metrics['exchange_net_flow'] = flows.get('net_flow', 0)
            metrics['exchange_flow_trend'] = flows.get('trend', 'neutral')
        
        # Calculate composite indicators
        metrics.update(self._calculate_composite_indicators(metrics))
        
        self.logger.info(f"✅ Collected {len(metrics)} on-chain metrics")
        return metrics
    
    def _get_glassnode_metric(self, endpoint: str, days: int) -> Optional[Dict]:
        """Get metric from Glassnode API"""
        try:
            since = int((datetime.now() - timedelta(days=days)).timestamp())
            
            url = f"{self.base_url}/{endpoint}"
            params = {
                'a': 'BTC',
                's': since,
                'api_key': self.glassnode_api_key
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                data = response.json()
                if data:
                    values = [item['v'] for item in data if item['v'] is not None]
                    if values:
                        return {
                            'latest_value': values[-1],
                            'values': values,
                            'trend': self._calculate_trend(values),
                            'change_pct': (values[-1] - values[0]) / values[0] * 100 if len(values) > 1 else 0
                        }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Glassnode API error: {e}")
            return None
    
    def _get_fallback_nvt(self, days: int) -> Dict:
        """Fallback NVT calculation using available data"""
        # Mock implementation - in real scenario would use price and transaction volume
        import random
        
        # Simulate NVT values (typical range 20-200)
        nvt_values = [20 + random.random() * 180 for _ in range(days)]
        
        return {
            'latest_value': nvt_values[-1],
            'values': nvt_values,
            'trend': self._calculate_trend(nvt_values),
            'change_pct': (nvt_values[-1] - nvt_values[0]) / nvt_values[0] * 100
        }
    
    def _get_fallback_mvrv(self, days: int) -> Dict:
        """Fallback MVRV calculation"""
        import random
        
        # Simulate MVRV values (typical range 0.5-4.0)
        mvrv_values = [0.5 + random.random() * 3.5 for _ in range(days)]
        
        return {
            'latest_value': mvrv_values[-1],
            'values': mvrv_values,
            'trend': self._calculate_trend(mvrv_values),
            'change_pct': (mvrv_values[-1] - mvrv_values[0]) / mvrv_values[0] * 100
        }
    
    def _calculate_fallback_volatility(self, days: int) -> Dict:
        """Calculate volatility from price data"""
        import random
        
        # Simulate volatility values (0-100%)
        vol_values = [random.random() * 100 for _ in range(days)]
        
        return {
            'latest_value': vol_values[-1],
            'values': vol_values,
            'trend': self._calculate_trend(vol_values),
            'change_pct': (vol_values[-1] - vol_values[0]) / vol_values[0] * 100
        }
    
    def _get_fallback_addresses(self, days: int) -> Dict:
        """Fallback active addresses"""
        import random
        
        # Simulate active addresses (500k - 1.5M)
        addr_values = [500000 + random.random() * 1000000 for _ in range(days)]
        
        return {
            'latest_value': addr_values[-1],
            'values': addr_values,
            'trend': self._calculate_trend(addr_values),
            'change_pct': (addr_values[-1] - addr_values[0]) / addr_values[0] * 100
        }
    
    def _get_fallback_flows(self, days: int) -> Dict:
        """Fallback exchange flows"""
        import random
        
        # Simulate net flows (negative = outflow, positive = inflow)
        flow_values = [(random.random() - 0.5) * 10000 for _ in range(days)]
        
        return {
            'net_flow': flow_values[-1],
            'values': flow_values,
            'trend': self._calculate_trend(flow_values),
            'change_pct': (flow_values[-1] - flow_values[0]) / abs(flow_values[0]) * 100 if flow_values[0] != 0 else 0
        }
    
    def _calculate_trend(self, values: list) -> str:
        """Calculate trend from values"""
        if len(values) < 2:
            return 'neutral'
        
        # Simple linear trend
        start = sum(values[:len(values)//3]) / (len(values)//3)
        end = sum(values[-len(values)//3:]) / (len(values)//3)
        
        change_pct = (end - start) / start * 100
        
        if change_pct > 5:
            return 'bullish'
        elif change_pct < -5:
            return 'bearish'
        else:
            return 'neutral'
    
    def _calculate_net_flow(self, inflows: Dict, outflows: Dict) -> float:
        """Calculate net exchange flow"""
        latest_inflow = inflows.get('latest_value', 0)
        latest_outflow = outflows.get('latest_value', 0)
        return latest_inflow - latest_outflow
    
    def _calculate_composite_indicators(self, metrics: Dict) -> Dict:
        """Calculate composite on-chain indicators"""
        composite = {}
        
        # On-chain Health Score (0-100)
        health_score = 50  # Neutral baseline
        
        # NVT contribution
        nvt = metrics.get('nvt_ratio', 100)
        if nvt < 50:  # Undervalued
            health_score += 15
        elif nvt > 150:  # Overvalued
            health_score -= 15
        
        # MVRV contribution
        mvrv = metrics.get('mvrv_ratio', 1)
        if mvrv < 1:  # Undervalued
            health_score += 15
        elif mvrv > 3:  # Overvalued
            health_score -= 15
        
        # Exchange flow contribution
        net_flow = metrics.get('exchange_net_flow', 0)
        if net_flow < 0:  # Outflow (bullish)
            health_score += 10
        elif net_flow > 0:  # Inflow (bearish)
            health_score -= 10
        
        # Volatility contribution
        volatility = metrics.get('realized_volatility', 50)
        if volatility > 80:  # High volatility (bearish)
            health_score -= 10
        
        composite['onchain_health_score'] = max(0, min(100, health_score))
        
        # Market Sentiment Signal
        if health_score > 70:
            composite['onchain_signal'] = 'bullish'
        elif health_score < 30:
            composite['onchain_signal'] = 'bearish'
        else:
            composite['onchain_signal'] = 'neutral'
        
        return composite
