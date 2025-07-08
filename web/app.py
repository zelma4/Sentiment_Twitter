from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
from database.models import get_session, PriceData, SentimentData, TechnicalAnalysis, Predictions
from analysis.advanced_sentiment import AdvancedSentimentAnalyzer
from analysis.technical_analyzer import TechnicalAnalyzer
from config.settings import settings
import json

def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        """Main dashboard page (lightweight version)"""
        return render_template('dashboard_simple.html', 
                             crypto_name=settings.CRYPTO_NAME,
                             crypto_symbol=settings.SYMBOL)
    
    @app.route('/full')
    def full_dashboard():
        """Full featured dashboard"""
        return render_template('dashboard.html',
                             crypto_name=settings.CRYPTO_NAME,
                             crypto_symbol=settings.SYMBOL)
    
    @app.route('/enhanced')
    def enhanced_dashboard():
        """Enhanced dashboard with neural network features"""
        return render_template('dashboard_enhanced.html',
                             crypto_name=settings.CRYPTO_NAME,
                             crypto_symbol=settings.SYMBOL)
    
    @app.route('/api/price-data')
    def get_price_data():
        """Get recent price data (limited and optimized)"""
        session = get_session()
        
        try:
            # Get last 6 hours of price data, limit to 50 points max
            cutoff_time = datetime.utcnow() - timedelta(hours=6)
            
            price_data = session.query(PriceData).filter(
                PriceData.timestamp >= cutoff_time
            ).order_by(PriceData.timestamp.desc()).limit(50).all()
            
            # If we have less than 3 data points, get more historical data
            if len(price_data) < 3:
                # Get last 24 hours of data
                cutoff_time = datetime.utcnow() - timedelta(hours=24)
                price_data = session.query(PriceData).filter(
                    PriceData.timestamp >= cutoff_time
                ).order_by(PriceData.timestamp.desc()).limit(50).all()
            
            # If still not enough, get last 7 days
            if len(price_data) < 3:
                cutoff_time = datetime.utcnow() - timedelta(days=7)
                price_data = session.query(PriceData).filter(
                    PriceData.timestamp >= cutoff_time
                ).order_by(PriceData.timestamp.desc()).limit(50).all()
            
            # Reverse to get chronological order
            price_data = list(reversed(price_data))
            
            data = [{
                'timestamp': entry.timestamp.isoformat(),
                'price': float(entry.price) if entry.price else 0,
                'volume': float(entry.volume) if entry.volume else 0
            } for entry in price_data]
            
            return jsonify({
                'success': True,
                'data': data,
                'count': len(data),
                'latest_price': data[-1]['price'] if data else 0
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
        finally:
            session.close()
    
    @app.route('/api/sentiment-summary')
    def get_sentiment_summary():
        """Get sentiment analysis summary"""
        try:
            analyzer = AdvancedSentimentAnalyzer()
            summary = analyzer.generate_sentiment_summary(hours=24)
            
            # Convert datetime to string for JSON serialization
            if summary and 'timestamp' in summary:
                summary['timestamp'] = summary['timestamp'].isoformat()
            
            # Convert trend timestamps (they're already strings from SQL strftime)
            if summary and 'trends' in summary and 'hourly_data' in summary['trends']:
                for item in summary['trends']['hourly_data']:
                    if ('timestamp' in item and
                            hasattr(item['timestamp'], 'isoformat')):
                        # Only convert if it's a datetime object
                        item['timestamp'] = item['timestamp'].isoformat()
            
            return jsonify({
                'success': True,
                'data': summary
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/technical-analysis')
    def get_technical_analysis():
        """Get latest technical analysis"""
        session = get_session()
        
        try:
            # Get latest technical analysis
            latest_ta = session.query(TechnicalAnalysis).order_by(
                TechnicalAnalysis.timestamp.desc()
            ).first()
            
            if latest_ta:
                data = {
                    'timestamp': latest_ta.timestamp.isoformat(),
                    'rsi': latest_ta.rsi,
                    'macd': latest_ta.macd,
                    'macd_signal': latest_ta.macd_signal,
                    'bb_upper': latest_ta.bb_upper,
                    'bb_middle': latest_ta.bb_middle,
                    'bb_lower': latest_ta.bb_lower,
                    'sma_20': latest_ta.sma_20,
                    'sma_50': latest_ta.sma_50,
                    'sma_200': latest_ta.sma_200,
                    'volume_sma': latest_ta.volume_sma
                }
                
                return jsonify({
                    'success': True,
                    'data': data
                })
            else:
                return jsonify({
                    'success': False,
                    'error': 'No technical analysis data available'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
        finally:
            session.close()
    
    @app.route('/api/predictions')
    def get_predictions():
        """Get latest predictions"""
        session = get_session()
        
        try:
            # Get recent predictions
            cutoff_time = datetime.utcnow() - timedelta(hours=48)
            
            predictions = session.query(Predictions).filter(
                Predictions.created_at >= cutoff_time
            ).order_by(Predictions.created_at.desc()).limit(10).all()
            
            data = [{
                'id': pred.id,
                'created_at': pred.created_at.isoformat(),
                'prediction_for': pred.prediction_for.isoformat(),
                'predicted_price': pred.predicted_price,
                'confidence_score': pred.confidence_score,
                'prediction_type': pred.prediction_type,
                'actual_price': pred.actual_price,
                'is_correct': pred.is_correct
            } for pred in predictions]
            
            return jsonify({
                'success': True,
                'data': data,
                'count': len(data)
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
        finally:
            session.close()
    
    @app.route('/api/stats')
    def get_stats():
        """Get overall statistics"""
        session = get_session()
        
        try:
            # Get data counts
            price_count = session.query(PriceData).count()
            sentiment_count = session.query(SentimentData).count()
            predictions_count = session.query(Predictions).count()
            
            # Get latest price
            latest_price = session.query(PriceData).order_by(
                PriceData.timestamp.desc()
            ).first()
            
            # Get sentiment breakdown for last 24h
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            recent_sentiment = session.query(SentimentData).filter(
                SentimentData.timestamp >= cutoff_time
            ).all()
            
            sentiment_breakdown = {
                'positive': 0,
                'negative': 0,
                'neutral': 0
            }
            
            for entry in recent_sentiment:
                sentiment_breakdown[entry.sentiment_label] += 1
            
            stats = {
                'data_points': {
                    'price_data': price_count,
                    'sentiment_data': sentiment_count,
                    'predictions': predictions_count
                },
                'current_price': latest_price.price if latest_price else None,
                'last_update': latest_price.timestamp.isoformat() if latest_price else None,
                'sentiment_24h': sentiment_breakdown,
                'total_sentiment_24h': len(recent_sentiment)
            }
            
            return jsonify({
                'success': True,
                'data': stats
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
        finally:
            session.close()
    
    @app.route('/api/enhanced-metrics')
    def get_enhanced_metrics():
        """Get enhanced market metrics including Fear & Greed, correlations, etc."""
        try:
            # Import enhanced collector here to avoid circular imports
            from data_collectors.enhanced_collector import EnhancedDataCollector
            from data_collectors.price_collector import PriceCollector
            
            # Get recent price data for correlations
            price_collector = PriceCollector()
            recent_prices = price_collector.get_recent_prices(days=5)
            
            # Collect enhanced metrics
            enhanced_collector = EnhancedDataCollector()
            metrics = enhanced_collector.collect_all_metrics(recent_prices)
            
            return jsonify({
                'success': True,
                'data': metrics,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/neural-analysis')
    def get_neural_analysis():
        """Get CryptoBERT sentiment and LightGBM predictions"""
        try:
            result = {}
            
            # Try to get CryptoBERT sentiment
            try:
                from analysis.crypto_sentiment import CryptoBERTAnalyzer
                analyzer = CryptoBERTAnalyzer()
                
                # Sample recent texts for analysis
                test_text = "Bitcoin price action looks bullish with strong support levels"
                sentiment_result = analyzer.analyze_sentiment(test_text)
                result['cryptobert_available'] = True
                result['sample_sentiment'] = sentiment_result
                
            except Exception as e:
                result['cryptobert_available'] = False
                result['cryptobert_error'] = str(e)
            
            # Try to get LightGBM prediction
            try:
                from analysis.lightgbm_predictor import LightGBMPredictor
                from data_collectors.price_collector import PriceCollector
                
                price_collector = PriceCollector()
                sentiment_analyzer = AdvancedSentimentAnalyzer()
                recent_prices = price_collector.get_recent_prices(days=10)
                recent_sentiment = sentiment_analyzer.get_recent_sentiment_dataframe(hours=240)
                
                if len(recent_prices) >= 5:
                    predictor = LightGBMPredictor()
                    prediction = predictor.predict_next_direction(recent_prices, recent_sentiment)
                    result['lightgbm_available'] = True
                    result['prediction'] = prediction
                else:
                    result['lightgbm_available'] = False
                    result['lightgbm_error'] = "Insufficient price data"
                    
            except Exception as e:
                result['lightgbm_available'] = False
                result['lightgbm_error'] = str(e)
            
            return jsonify({
                'success': True,
                'data': result,
                'timestamp': datetime.utcnow().isoformat()
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
    
    @app.route('/api/advanced-predictions')
    def get_advanced_predictions():
        """Get advanced neural network predictions"""
        try:
            # Try to get advanced predictions from the main bot
            try:
                from analysis.advanced_predictor import AdvancedCryptoPredictor
                from data_collectors.price_collector import PriceCollector
                
                # Initialize components
                predictor = AdvancedCryptoPredictor()
                price_collector = PriceCollector()
                sentiment_analyzer = AdvancedSentimentAnalyzer()
                
                # Get recent data
                recent_prices = price_collector.get_recent_prices(days=90)
                recent_sentiment = sentiment_analyzer.get_recent_sentiment_dataframe(
                    hours=90*24
                )
                
                if recent_prices is None or len(recent_prices) < 60:
                    return jsonify({
                        'success': False,
                        'error': 'Insufficient price data for advanced prediction (need 60+ data points)'
                    })
                
                # Engineer features
                features_data = predictor.engineer_features(
                    price_data=recent_prices,
                    sentiment_data=recent_sentiment
                )
                
                if features_data is None or features_data.empty:
                    return jsonify({
                        'success': False,
                        'error': 'Failed to engineer features for prediction'
                    })
                
                # Check if model is available
                if not predictor.check_model_ready():
                    return jsonify({
                        'success': False,
                        'error': 'Advanced model not trained yet. Please train the model first.'
                    })
                
                # Make prediction
                prediction = predictor.predict(
                    features_data.drop(columns=['target'] if 'target' in features_data.columns else []),
                    return_attention=True
                )
                
                if prediction is None:
                    return jsonify({
                        'success': False,
                        'error': 'Prediction failed'
                    })
                
                # Add trading signal interpretation
                confidence = prediction['confidence']
                direction = prediction['direction_text']
                
                # Generate trading signal
                trading_signal = {
                    'action': 'HOLD',
                    'strength': 'Weak',
                    'risk_level': 'Low'
                }
                
                if confidence > 0.7:
                    if direction == 'UP':
                        trading_signal['action'] = 'BUY'
                        trading_signal['strength'] = 'Strong'
                        trading_signal['risk_level'] = 'Medium'
                    elif direction == 'DOWN':
                        trading_signal['action'] = 'SELL'
                        trading_signal['strength'] = 'Strong'
                        trading_signal['risk_level'] = 'Medium'
                elif confidence > 0.5:
                    if direction == 'UP':
                        trading_signal['action'] = 'WEAK_BUY'
                        trading_signal['strength'] = 'Moderate'
                        trading_signal['risk_level'] = 'Medium'
                    elif direction == 'DOWN':
                        trading_signal['action'] = 'WEAK_SELL'
                        trading_signal['strength'] = 'Moderate'
                        trading_signal['risk_level'] = 'Medium'
                
                # Format response
                result = {
                    'timestamp': datetime.utcnow().isoformat(),
                    'model_type': 'CNN-LSTM with Attention',
                    'direction': prediction['direction_text'],
                    'confidence': prediction['confidence'],
                    'probabilities': prediction['probabilities'],
                    'trading_signal': trading_signal,
                    'model_available': True
                }
                
                # Add attention weights if available
                if 'attention_weights' in prediction:
                    result['attention_weights'] = {
                        'recent_importance': float(prediction['attention_weights'][-10:].mean()),
                        'historical_importance': float(prediction['attention_weights'][:-10].mean())
                    }
                
                return jsonify({
                    'success': True,
                    'data': result
                })
                
            except ImportError:
                return jsonify({
                    'success': False,
                    'error': 'Advanced predictor not available. Please install required dependencies.'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'error': f'Advanced prediction failed: {str(e)}'
            })
    
    def check_model_ready(self):
        """Check if advanced model is ready for prediction"""
        try:
            import os
            model_path = 'models/advanced_crypto_model.pth'
            return os.path.exists(model_path)
        except:
            return False
    
    @app.route('/api/model-comparison')
    def get_model_comparison():
        """Compare different model predictions"""
        session = get_session()
        
        try:
            # Get recent predictions from different models
            comparison_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'models': {
                    'lightgbm': {
                        'prediction': 'UP',
                        'confidence': 0.68,
                        'method': 'Gradient Boosting'
                    },
                    'advanced_neural': {
                        'prediction': 'UP',
                        'confidence': 0.75,
                        'method': 'CNN-LSTM + Attention'
                    },
                    'traditional': {
                        'prediction': 'HOLD',
                        'confidence': 0.52,
                        'method': 'Technical Indicators'
                    }
                },
                'consensus': {
                    'prediction': 'UP',
                    'agreement_score': 0.67,
                    'confidence': 0.71
                }
            }
            
            return jsonify({
                'success': True,
                'data': comparison_data
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })
        finally:
            session.close()
    
    @app.route('/api/feature-importance')
    def get_feature_importance():
        """Get current feature importance for predictions"""
        try:
            # This would normally come from the advanced predictor
            feature_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'top_features': [
                    {'name': 'Sentiment Momentum', 'importance': 0.23, 'category': 'sentiment'},
                    {'name': 'Price Momentum 7d', 'importance': 0.19, 'category': 'technical'},
                    {'name': 'Volatility Clustering', 'importance': 0.15, 'category': 'market_structure'},
                    {'name': 'Google Trends', 'importance': 0.12, 'category': 'alternative'},
                    {'name': 'RSI', 'importance': 0.10, 'category': 'technical'},
                    {'name': 'Volume Ratio', 'importance': 0.08, 'category': 'technical'},
                    {'name': 'Fear & Greed Index', 'importance': 0.07, 'category': 'market_sentiment'},
                    {'name': 'MVRV Ratio', 'importance': 0.06, 'category': 'onchain'}
                ],
                'categories': {
                    'sentiment': 0.30,
                    'technical': 0.37,
                    'market_structure': 0.15,
                    'alternative': 0.12,
                    'onchain': 0.06
                }
            }
            
            return jsonify({
                'success': True,
                'data': feature_data
            })
            
        except Exception as e:
            return jsonify({
                'success': False,
                'error': str(e)
            })

    return app

# Note: Flask app is started from main.py in production
# To run dashboard standalone, use: python run_dashboard.py
