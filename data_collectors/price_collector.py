import ccxt
import yfinance as yf
import requests
import logging
from datetime import datetime, timedelta
from config.settings import settings
from database.models import get_session, PriceData
import pandas as pd

class PriceCollector:
    def __init__(self):
        self.setup_logging()
        self.binance = self.setup_binance()
        self.coingecko_url = "https://api.coingecko.com/api/v3"
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def setup_binance(self):
        """Setup Binance exchange connection"""
        try:
            exchange = ccxt.binance({
                'apiKey': settings.BINANCE_API_KEY,
                'secret': settings.BINANCE_SECRET_KEY,
                'sandbox': False,  # Set to True for testnet
                'enableRateLimit': True,
            })
            
            # Test connection
            try:
                exchange.load_markets()
                self.logger.info("Binance connection successful")
                return exchange
            except Exception as e:
                self.logger.warning(f"Binance API test failed: {e}")
                # Continue without API keys (public data only)
                exchange = ccxt.binance({'enableRateLimit': True})
                exchange.load_markets()
                return exchange
                
        except Exception as e:
            self.logger.error(f"Failed to setup Binance: {e}")
            return None
    
    def get_current_price_binance(self):
        """Get current Bitcoin price from Binance"""
        if not self.binance:
            return None
            
        try:
            ticker = self.binance.fetch_ticker(settings.TRADING_PAIR)
            
            price_data = {
                'price': ticker['last'],
                'volume': ticker['baseVolume'],
                'high': ticker['high'],
                'low': ticker['low'],
                'open_price': ticker['open'],
                'close_price': ticker['close'],
                'timestamp': datetime.utcnow()
            }
            
            self.logger.info(f"Binance {settings.SYMBOL} price: ${price_data['price']:.2f}")
            return price_data
            
        except Exception as e:
            self.logger.error(f"Error fetching Binance price: {e}")
            return None
    
    def get_current_price_coingecko(self):
        """Get current crypto price from CoinGecko"""
        try:
            coin_id = settings.COINGECKO_ID
            
            url = f"{self.coingecko_url}/simple/price"
            params = {
                'ids': coin_id,
                'vs_currencies': 'usd',
                'include_market_cap': 'true',
                'include_24hr_vol': 'true',
                'include_24hr_change': 'true'
            }
            
            if settings.COINGECKO_API_KEY:
                params['x_cg_demo_api_key'] = settings.COINGECKO_API_KEY
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()[coin_id]
            
            price_data = {
                'price': data['usd'],
                'market_cap': data.get('usd_market_cap'),
                'volume': data.get('usd_24h_vol'),
                'timestamp': datetime.utcnow()
            }
            
            self.logger.info(f"CoinGecko {settings.SYMBOL} price: ${price_data['price']:.2f}")
            return price_data
            
        except Exception as e:
            self.logger.error(f"Error fetching CoinGecko price: {e}")
            return None
    
    def get_historical_data_binance(self, days=30):
        """Get historical price data from Binance"""
        if not self.binance:
            return None
            
        try:
            # Calculate timestamp for 'days' ago
            since = int((datetime.utcnow() - timedelta(days=days)).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.binance.fetch_ohlcv(
                settings.TRADING_PAIR, 
                timeframe='1d',
                since=since,
                limit=days
            )
            
            historical_data = []
            for candle in ohlcv:
                price_data = {
                    'timestamp': datetime.fromtimestamp(candle[0] / 1000),
                    'open_price': candle[1],
                    'high': candle[2],
                    'low': candle[3],
                    'close_price': candle[4],
                    'price': candle[4],  # Use close price as current price
                    'volume': candle[5]
                }
                historical_data.append(price_data)
            
            self.logger.info(f"Fetched {len(historical_data)} days of historical data from Binance")
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data from Binance: {e}")
            return None
    
    def get_historical_data_yfinance(self, days=30):
        """Get historical price data from Yahoo Finance"""
        try:
            # Download Bitcoin data
            btc = yf.Ticker("BTC-USD")
            hist = btc.history(period=f"{days}d")
            
            historical_data = []
            for date, row in hist.iterrows():
                price_data = {
                    'timestamp': date.to_pydatetime().replace(tzinfo=None),
                    'open_price': row['Open'],
                    'high': row['High'],
                    'low': row['Low'],
                    'close_price': row['Close'],
                    'price': row['Close'],
                    'volume': row['Volume']
                }
                historical_data.append(price_data)
            
            self.logger.info(f"Fetched {len(historical_data)} days of historical data from Yahoo Finance")
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data from Yahoo Finance: {e}")
            return None
    
    def get_market_data_coingecko(self, days=30):
        """Get comprehensive market data from CoinGecko"""
        try:
            coin_id = settings.COINGECKO_ID
            
            url = f"{self.coingecko_url}/coins/{coin_id}/market_chart"
            params = {
                'vs_currency': 'usd',
                'days': days,
                'interval': 'daily'
            }
            
            if settings.COINGECKO_API_KEY:
                params['x_cg_demo_api_key'] = settings.COINGECKO_API_KEY
            
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            # Combine price, market cap, and volume data
            prices = data['prices']
            market_caps = data['market_caps']
            volumes = data['total_volumes']
            
            historical_data = []
            for i in range(len(prices)):
                price_data = {
                    'timestamp': datetime.fromtimestamp(prices[i][0] / 1000),
                    'price': prices[i][1],
                    'close_price': prices[i][1],
                    'market_cap': market_caps[i][1] if i < len(market_caps) else None,
                    'volume': volumes[i][1] if i < len(volumes) else None
                }
                historical_data.append(price_data)
            
            self.logger.info(f"Fetched {len(historical_data)} days of market data from CoinGecko")
            return historical_data
            
        except Exception as e:
            self.logger.error(f"Error fetching market data from CoinGecko: {e}")
            return None
    
    def save_to_database(self, price_data_list):
        """Save price data to database"""
        session = get_session()
        saved_count = 0
        
        try:
            for price_data in price_data_list:
                # Check if data for this timestamp already exists
                existing = session.query(PriceData).filter(
                    PriceData.timestamp >= price_data['timestamp'] - timedelta(minutes=30),
                    PriceData.timestamp <= price_data['timestamp'] + timedelta(minutes=30)
                ).first()
                
                if not existing:
                    price_entry = PriceData(
                        timestamp=price_data['timestamp'],
                        symbol=settings.SYMBOL,
                        price=price_data['price'],
                        volume=price_data.get('volume'),
                        high=price_data.get('high'),
                        low=price_data.get('low'),
                        open_price=price_data.get('open_price'),
                        close_price=price_data.get('close_price'),
                        market_cap=price_data.get('market_cap')
                    )
                    
                    session.add(price_entry)
                    saved_count += 1
            
            session.commit()
            self.logger.info(f"Saved {saved_count} new price entries to database")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving price data to database: {e}")
        finally:
            session.close()
            
        return saved_count
    
    def collect_current_data(self):
        """Collect current price data from multiple sources"""
        current_data = []
        
        # Try Binance first
        binance_data = self.get_current_price_binance()
        if binance_data:
            current_data.append(binance_data)
        
        # Try CoinGecko as backup
        if not binance_data:
            coingecko_data = self.get_current_price_coingecko()
            if coingecko_data:
                current_data.append(coingecko_data)
        
        return current_data
    
    def collect_historical_data(self, days=30):
        """Collect historical price data"""
        historical_data = None
        
        # Try Binance first
        historical_data = self.get_historical_data_binance(days)
        
        # Try Yahoo Finance as backup
        if not historical_data:
            historical_data = self.get_historical_data_yfinance(days)
        
        # Try CoinGecko as last resort
        if not historical_data:
            historical_data = self.get_market_data_coingecko(days)
        
        return historical_data or []
    
    def collect_and_save(self, include_historical=False, historical_days=30):
        """Main method to collect and save price data"""
        self.logger.info("Starting price data collection...")
        
        all_data = []
        
        # Collect current data
        current_data = self.collect_current_data()
        all_data.extend(current_data)
        
        # Collect historical data if requested
        if include_historical:
            historical_data = self.collect_historical_data(historical_days)
            all_data.extend(historical_data)
        
        # Save to database
        if all_data:
            saved_count = self.save_to_database(all_data)
            
            stats = {
                'total_entries': len(all_data),
                'current_data': len(current_data),
                'historical_data': len(all_data) - len(current_data),
                'saved_entries': saved_count,
                'latest_price': all_data[-1]['price'] if all_data else None
            }
            
            self.logger.info(f"Price collection complete: {stats}")
            return stats
        else:
            self.logger.warning("No price data collected")
            return None
    
    def get_recent_prices(self, days=30):
        """Get recent price data as DataFrame"""
        session = get_session()
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(days=days)
            
            price_data = session.query(PriceData).filter(
                PriceData.timestamp >= cutoff_time
            ).order_by(PriceData.timestamp.asc()).all()
            
            if not price_data:
                self.logger.warning(f"No price data found for last {days} days")
                return pd.DataFrame()
            
            # Convert to DataFrame
            data = []
            for entry in price_data:
                data.append({
                    'date': entry.timestamp.date(),
                    'open': entry.open_price or entry.price,
                    'high': entry.high or entry.price,
                    'low': entry.low or entry.price,
                    'close': entry.close_price or entry.price,
                    'volume': entry.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('date', inplace=True)
            df = df.sort_index()
            
            self.logger.info(f"Retrieved {len(df)} days of price data")
            return df
            
        except Exception as e:
            self.logger.error(f"Error getting recent prices: {e}")
            return pd.DataFrame()
        finally:
            session.close()
    
    def get_current_price(self):
        """Get current crypto price (tries Binance first, then CoinGecko)"""
        try:
            # Try Binance first
            binance_data = self.get_current_price_binance()
            if binance_data:
                return binance_data['price']
            
            # Fallback to CoinGecko
            coingecko_data = self.get_current_price_coingecko()
            if coingecko_data:
                return coingecko_data['price']
                
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get current price: {e}")
            return None

if __name__ == "__main__":
    collector = PriceCollector()
    # Collect current data
    collector.collect_and_save()
    # Collect historical data (uncomment for initial setup)
    # collector.collect_and_save(include_historical=True, historical_days=30)
