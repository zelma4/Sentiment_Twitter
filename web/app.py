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
        """Main dashboard page"""
        return render_template('dashboard.html')
    
    @app.route('/api/price-data')
    def get_price_data():
        """Get recent price data"""
        session = get_session()
        
        try:
            # Get last 24 hours of price data
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            price_data = session.query(PriceData).filter(
                PriceData.timestamp >= cutoff_time
            ).order_by(PriceData.timestamp).all()
            
            data = [{
                'timestamp': entry.timestamp.isoformat(),
                'price': entry.price,
                'volume': entry.volume or 0
            } for entry in price_data]
            
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
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=8000)
