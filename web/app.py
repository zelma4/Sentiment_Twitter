from flask import Flask, render_template, jsonify
from datetime import datetime, timedelta
from database.models import get_session, PriceData, SentimentData, TechnicalAnalysis, Predictions
from analysis.sentiment_analyzer import SentimentAnalyzer
from analysis.technical_analyzer import TechnicalAnalyzer
import json

def create_app():
    app = Flask(__name__)
    
    @app.route('/')
    def dashboard():
        """Main dashboard page (lightweight version)"""
        return render_template('dashboard_simple.html')
    
    @app.route('/full')
    def full_dashboard():
        """Full featured dashboard"""
        return render_template('dashboard.html')
    
    @app.route('/enhanced')
    def enhanced_dashboard():
        """Enhanced dashboard with neural network features"""
        return render_template('dashboard_enhanced.html')
    
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
            analyzer = SentimentAnalyzer()
            summary = analyzer.generate_sentiment_summary(hours=24)
            
            # Convert datetime to string for JSON serialization
            if summary and 'timestamp' in summary:
                summary['timestamp'] = summary['timestamp'].isoformat()
            
            # Convert trend timestamps
            if summary and 'trends' in summary and 'hourly_data' in summary['trends']:
                for item in summary['trends']['hourly_data']:
                    if 'timestamp' in item:
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
                recent_prices = price_collector.get_recent_prices(days=10)
                
                if len(recent_prices) >= 5:
                    predictor = LightGBMPredictor()
                    prediction = predictor.predict_next_direction(recent_prices)
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

    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8000)
