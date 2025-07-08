"""
Simple Backtesting Framework for Bitcoin Trading Strategies
Tests model predictions against historical data for PnL analysis
"""

import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

class SimpleBTCBacktester:
    """
    Simple backtesting framework for Bitcoin trading strategies
    Focuses on PnL rather than just accuracy metrics
    """
    
    def __init__(self, initial_balance=10000.0, transaction_fee=0.001):
        self.setup_logging()
        self.initial_balance = initial_balance
        self.transaction_fee = transaction_fee  # 0.1% fee
        self.balance = initial_balance
        self.btc_holdings = 0.0
        self.trade_history = []
        self.equity_curve = []
        
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
    
    def run_backtest(self, price_data, predictions, confidence_threshold=0.6):
        """
        Run backtest with price data and model predictions
        
        Args:
            price_data: DataFrame with columns ['timestamp', 'close', 'volume']
            predictions: List of prediction dictionaries
            confidence_threshold: Minimum confidence to act on signal
            
        Returns:
            Dictionary with backtest results
        """
        try:
            self.logger.info("🔄 Starting backtest...")
            
            # Reset state
            self.balance = self.initial_balance
            self.btc_holdings = 0.0
            self.trade_history = []
            self.equity_curve = []
            
            # Align predictions with price data
            aligned_data = self._align_predictions_with_prices(price_data, predictions)
            
            for i, row in aligned_data.iterrows():
                price = row['close']
                prediction = row.get('prediction')
                
                # Calculate current equity
                total_equity = self.balance + (self.btc_holdings * price)
                self.equity_curve.append({
                    'timestamp': row['timestamp'],
                    'equity': total_equity,
                    'btc_price': price,
                    'btc_holdings': self.btc_holdings,
                    'cash_balance': self.balance
                })
                
                # Make trading decision if we have a prediction
                if prediction and prediction.get('confidence', 0) >= confidence_threshold:
                    self._execute_trade(
                        price=price,
                        signal=prediction['direction_text'],
                        confidence=prediction['confidence'],
                        timestamp=row['timestamp']
                    )
            
            # Calculate final results
            final_price = aligned_data.iloc[-1]['close']
            final_equity = self.balance + (self.btc_holdings * final_price)
            
            results = self._calculate_performance_metrics(final_equity, final_price)
            
            self.logger.info(f"✅ Backtest complete. Final PnL: {results['total_return']:.2f}%")
            
            return results
            
        except Exception as e:
            self.logger.error(f"Backtest error: {e}")
            return {}
    
    def _align_predictions_with_prices(self, price_data, predictions):
        """Align predictions with price data by timestamp"""
        try:
            # Convert predictions to DataFrame
            pred_df = pd.DataFrame(predictions)
            if 'timestamp' not in pred_df.columns:
                # If no timestamp, assume chronological order
                pred_df['timestamp'] = pd.date_range(
                    start=price_data.iloc[0]['timestamp'],
                    periods=len(predictions),
                    freq='H'
                )
            
            # Merge with price data
            price_df = price_data.copy()
            price_df['timestamp'] = pd.to_datetime(price_df['timestamp'])
            pred_df['timestamp'] = pd.to_datetime(pred_df['timestamp'])
            
            # Merge on timestamp (forward fill predictions)
            merged = pd.merge_asof(
                price_df.sort_values('timestamp'),
                pred_df.sort_values('timestamp'),
                on='timestamp',
                direction='backward'
            )
            
            return merged
            
        except Exception as e:
            self.logger.error(f"Error aligning data: {e}")
            return price_data
    
    def _execute_trade(self, price, signal, confidence, timestamp):
        """Execute a trade based on signal"""
        try:
            # Simple strategy: BUY on UP signal, SELL on DOWN signal, HOLD otherwise
            if signal == 'UP' and self.balance > 100:  # Have cash to buy
                # Buy BTC with available cash
                trade_amount = self.balance * 0.95  # Use 95% of cash (keep some for fees)
                btc_amount = trade_amount / price
                fee = trade_amount * self.transaction_fee
                
                self.btc_holdings += btc_amount
                self.balance -= (trade_amount + fee)
                
                self.trade_history.append({
                    'timestamp': timestamp,
                    'action': 'BUY',
                    'price': price,
                    'amount': btc_amount,
                    'value': trade_amount,
                    'fee': fee,
                    'confidence': confidence,
                    'balance_after': self.balance,
                    'btc_after': self.btc_holdings
                })
                
            elif signal == 'DOWN' and self.btc_holdings > 0.001:  # Have BTC to sell
                # Sell all BTC
                trade_value = self.btc_holdings * price
                fee = trade_value * self.transaction_fee
                
                self.balance += (trade_value - fee)
                
                self.trade_history.append({
                    'timestamp': timestamp,
                    'action': 'SELL',
                    'price': price,
                    'amount': self.btc_holdings,
                    'value': trade_value,
                    'fee': fee,
                    'confidence': confidence,
                    'balance_after': self.balance,
                    'btc_after': 0.0
                })
                
                self.btc_holdings = 0.0
                
        except Exception as e:
            self.logger.error(f"Trade execution error: {e}")
    
    def _calculate_performance_metrics(self, final_equity, final_price):
        """Calculate performance metrics"""
        try:
            # Basic metrics
            total_return = ((final_equity - self.initial_balance) / self.initial_balance) * 100
            
            # Buy and hold comparison
            initial_btc = self.initial_balance / self.equity_curve[0]['btc_price']
            buy_hold_final = initial_btc * final_price
            buy_hold_return = ((buy_hold_final - self.initial_balance) / self.initial_balance) * 100
            
            # Trade statistics
            num_trades = len(self.trade_history)
            buy_trades = [t for t in self.trade_history if t['action'] == 'BUY']
            sell_trades = [t for t in self.trade_history if t['action'] == 'SELL']
            
            # Calculate win rate (simplified)
            profitable_trades = 0
            total_completed_trades = 0
            
            for i in range(min(len(buy_trades), len(sell_trades))):
                buy_price = buy_trades[i]['price']
                sell_price = sell_trades[i]['price']
                if sell_price > buy_price:
                    profitable_trades += 1
                total_completed_trades += 1
            
            win_rate = (profitable_trades / total_completed_trades * 100) if total_completed_trades > 0 else 0
            
            # Volatility and Sharpe (simplified)
            equity_values = [e['equity'] for e in self.equity_curve]
            equity_returns = np.diff(equity_values) / equity_values[:-1]
            volatility = np.std(equity_returns) * np.sqrt(365 * 24)  # Annualized
            sharpe_ratio = (total_return / 100) / volatility if volatility > 0 else 0
            
            # Maximum drawdown
            peak = self.initial_balance
            max_drawdown = 0
            for equity in equity_values:
                if equity > peak:
                    peak = equity
                drawdown = (peak - equity) / peak
                if drawdown > max_drawdown:
                    max_drawdown = drawdown
            
            results = {
                'total_return': total_return,
                'buy_hold_return': buy_hold_return,
                'excess_return': total_return - buy_hold_return,
                'final_equity': final_equity,
                'initial_balance': self.initial_balance,
                'num_trades': num_trades,
                'win_rate': win_rate,
                'volatility': volatility * 100,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown * 100,
                'total_fees': sum(t['fee'] for t in self.trade_history),
                'profitable_trades': profitable_trades,
                'total_completed_trades': total_completed_trades
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def generate_report(self, results):
        """Generate a readable backtest report"""
        try:
            if not results:
                return "No backtest results available"
            
            report = f"""
📊 BITCOIN TRADING BACKTEST REPORT
{'=' * 45}

💰 PERFORMANCE METRICS:
  Initial Balance:    ${results['initial_balance']:,.2f}
  Final Equity:       ${results['final_equity']:,.2f}
  Total Return:       {results['total_return']:+.2f}%
  Buy & Hold Return:  {results['buy_hold_return']:+.2f}%
  Excess Return:      {results['excess_return']:+.2f}%

📈 TRADE STATISTICS:
  Total Trades:       {results['num_trades']}
  Completed Trades:   {results['total_completed_trades']}
  Win Rate:           {results['win_rate']:.1f}%
  Profitable Trades:  {results['profitable_trades']}

📊 RISK METRICS:
  Volatility:         {results['volatility']:.2f}%
  Sharpe Ratio:       {results['sharpe_ratio']:.2f}
  Max Drawdown:       {results['max_drawdown']:.2f}%
  Total Fees:         ${results['total_fees']:.2f}

🎯 CONCLUSION:
  Strategy {'OUTPERFORMED' if results['excess_return'] > 0 else 'UNDERPERFORMED'} Buy & Hold by {abs(results['excess_return']):.2f}%
  Risk-Adjusted Return (Sharpe): {results['sharpe_ratio']:.2f}
"""
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            return "Error generating backtest report"
    
    def plot_equity_curve(self, save_path=None):
        """Plot equity curve vs BTC price"""
        try:
            if not self.equity_curve:
                self.logger.warning("No equity curve data to plot")
                return
            
            # Create DataFrame from equity curve
            df = pd.DataFrame(self.equity_curve)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Normalize both to starting value of 100
            initial_equity = df.iloc[0]['equity']
            initial_btc_price = df.iloc[0]['btc_price']
            
            df['normalized_equity'] = (df['equity'] / initial_equity) * 100
            df['normalized_btc'] = (df['btc_price'] / initial_btc_price) * 100
            
            # Plot
            plt.figure(figsize=(12, 6))
            plt.plot(df['timestamp'], df['normalized_equity'], label='Strategy Equity', linewidth=2)
            plt.plot(df['timestamp'], df['normalized_btc'], label='BTC Buy & Hold', linewidth=2, alpha=0.7)
            
            plt.title('Strategy Performance vs Buy & Hold (Normalized to 100)')
            plt.xlabel('Date')
            plt.ylabel('Normalized Value')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Equity curve saved to {save_path}")
            else:
                plt.show()
                
        except Exception as e:
            self.logger.error(f"Error plotting equity curve: {e}")

def run_strategy_backtest(price_data, model_predictions, strategy_config=None):
    """
    Convenience function to run a complete backtest
    
    Args:
        price_data: Historical price data
        model_predictions: Model predictions with timestamps
        strategy_config: Configuration dictionary
        
    Returns:
        Tuple of (results, report_text)
    """
    try:
        # Default configuration
        config = {
            'initial_balance': 10000.0,
            'transaction_fee': 0.001,
            'confidence_threshold': 0.6
        }
        
        if strategy_config:
            config.update(strategy_config)
        
        # Run backtest
        backtester = SimpleBTCBacktester(
            initial_balance=config['initial_balance'],
            transaction_fee=config['transaction_fee']
        )
        
        results = backtester.run_backtest(
            price_data=price_data,
            predictions=model_predictions,
            confidence_threshold=config['confidence_threshold']
        )
        
        # Generate report
        report = backtester.generate_report(results)
        
        return results, report, backtester
        
    except Exception as e:
        logging.error(f"Backtest run error: {e}")
        return {}, "Backtest failed", None
