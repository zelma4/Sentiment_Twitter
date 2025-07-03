import logging
from datetime import datetime, timedelta
from sqlalchemy import func
from database.models import get_session, SentimentData
from config.settings import settings
import statistics

# Try to import advanced sentiment analyzers
try:
    from analysis.crypto_sentiment import CryptoBERTAnalyzer, HybridSentimentAnalyzer
    ADVANCED_SENTIMENT_AVAILABLE = True
except ImportError:
    ADVANCED_SENTIMENT_AVAILABLE = False

class SentimentAnalyzer:
    def __init__(self):
        self.setup_logging()
        
        # Initialize advanced sentiment analyzers if available
        if ADVANCED_SENTIMENT_AVAILABLE:
            try:
                self.cryptobert_analyzer = CryptoBERTAnalyzer()
                self.hybrid_analyzer = HybridSentimentAnalyzer()
                self.use_advanced = True
                self.logger.info("✅ Advanced CryptoBERT sentiment analysis enabled")
            except Exception as e:
                self.logger.warning(f"⚠️ Failed to initialize CryptoBERT: {e}")
                self.use_advanced = False
        else:
            self.use_advanced = False
            self.logger.info("📊 Using traditional sentiment analysis only")
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def analyze_text_sentiment(self, text, source="unknown"):
        """
        Analyze sentiment of a single text using best available method
        
        Args:
            text: Text to analyze
            source: Source of the text (twitter, reddit, news)
            
        Returns:
            Dict with sentiment score and metadata
        """
        if not text or not text.strip():
            return {
                'score': 0.0,
                'confidence': 0.0,
                'method': 'empty_text',
                'source': source
            }
        
        # Use CryptoBERT if available (best for crypto content)
        if self.use_advanced and source in ['twitter', 'reddit', 'social']:
            try:
                result = self.cryptobert_analyzer.analyze_sentiment(text)
                return {
                    'score': result['score'],
                    'confidence': result['confidence'],
                    'method': 'cryptobert',
                    'label': result['label'],
                    'source': source
                }
            except Exception as e:
                self.logger.warning(f"CryptoBERT failed, falling back: {e}")
        
        # Use hybrid analyzer for news and mixed content
        if self.use_advanced:
            try:
                result = self.hybrid_analyzer.analyze_comprehensive(text)
                return {
                    'score': result['hybrid_score'],
                    'confidence': result['primary_confidence'],
                    'method': 'hybrid',
                    'primary_label': result['primary_label'],
                    'source': source
                }
            except Exception as e:
                self.logger.warning(f"Hybrid analyzer failed, falling back: {e}")
        
        # Fallback to traditional methods
        return self._traditional_sentiment_analysis(text, source)
    
    def _traditional_sentiment_analysis(self, text, source):
        """Traditional sentiment analysis using VADER/TextBlob"""
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            
            analyzer = SentimentIntensityAnalyzer()
            scores = analyzer.polarity_scores(text)
            
            return {
                'score': scores['compound'],
                'confidence': abs(scores['compound']),
                'method': 'vader',
                'source': source,
                'raw_scores': scores
            }
        except Exception as e:
            self.logger.error(f"Traditional sentiment analysis failed: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'method': 'fallback',
                'source': source
            }
    
    def get_recent_sentiment_data(self, hours=24):
        """Get sentiment data from the last N hours"""
        session = get_session()
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            recent_data = session.query(SentimentData).filter(
                SentimentData.timestamp >= cutoff_time
            ).all()
            
            self.logger.info(f"Retrieved {len(recent_data)} sentiment entries from last {hours} hours")
            return recent_data
            
        except Exception as e:
            self.logger.error(f"Error retrieving sentiment data: {e}")
            return []
        finally:
            session.close()
    
    def calculate_overall_sentiment(self, hours=24):
        """Calculate overall sentiment metrics"""
        recent_data = self.get_recent_sentiment_data(hours)
        
        if not recent_data:
            return {
                'overall_score': 0.0,
                'confidence': 0.0,
                'total_posts': 0,
                'positive_ratio': 0.0,
                'negative_ratio': 0.0,
                'neutral_ratio': 0.0
            }
        
        # Calculate basic metrics
        scores = [entry.sentiment_score for entry in recent_data]
        labels = [entry.sentiment_label for entry in recent_data]
        
        total_posts = len(recent_data)
        positive_count = labels.count('positive')
        negative_count = labels.count('negative')
        neutral_count = labels.count('neutral')
        
        # Calculate ratios
        positive_ratio = positive_count / total_posts
        negative_ratio = negative_count / total_posts
        neutral_ratio = neutral_count / total_posts
        
        # Calculate weighted sentiment score
        # Give more weight to posts with higher engagement
        weighted_scores = []
        total_weight = 0
        
        for entry in recent_data:
            # Weight based on likes + retweets + replies
            engagement = entry.likes + entry.retweets + entry.replies
            weight = max(1, engagement)  # Minimum weight of 1
            
            weighted_scores.append(entry.sentiment_score * weight)
            total_weight += weight
        
        overall_score = sum(weighted_scores) / total_weight if total_weight > 0 else statistics.mean(scores)
        
        # Calculate confidence based on data quantity and consistency
        confidence = min(1.0, total_posts / settings.MIN_SENTIMENT_POSTS)
        
        # Reduce confidence if sentiment is very mixed
        score_std = statistics.stdev(scores) if len(scores) > 1 else 0
        consistency_factor = max(0.3, 1.0 - score_std)
        confidence *= consistency_factor
        
        return {
            'overall_score': overall_score,
            'confidence': confidence,
            'total_posts': total_posts,
            'positive_ratio': positive_ratio,
            'negative_ratio': negative_ratio,
            'neutral_ratio': neutral_ratio,
            'score_std': score_std,
            'avg_engagement': statistics.mean([e.likes + e.retweets + e.replies for e in recent_data])
        }
    
    def get_sentiment_by_source(self, hours=24):
        """Get sentiment breakdown by source (Twitter vs Reddit)"""
        recent_data = self.get_recent_sentiment_data(hours)
        
        twitter_data = [entry for entry in recent_data if entry.source == 'twitter']
        reddit_data = [entry for entry in recent_data if entry.source == 'reddit']
        
        def calculate_source_metrics(data):
            if not data:
                return {
                    'count': 0,
                    'avg_sentiment': 0.0,
                    'positive_ratio': 0.0,
                    'negative_ratio': 0.0,
                    'neutral_ratio': 0.0
                }
            
            scores = [entry.sentiment_score for entry in data]
            labels = [entry.sentiment_label for entry in data]
            
            return {
                'count': len(data),
                'avg_sentiment': statistics.mean(scores),
                'positive_ratio': labels.count('positive') / len(data),
                'negative_ratio': labels.count('negative') / len(data),
                'neutral_ratio': labels.count('neutral') / len(data)
            }
        
        return {
            'twitter': calculate_source_metrics(twitter_data),
            'reddit': calculate_source_metrics(reddit_data)
        }
    
    def get_sentiment_trends(self, hours=72):
        """Analyze sentiment trends over time"""
        session = get_session()
        
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            
            # Group data by hour (SQLite compatible)
            hourly_data = session.query(
                func.strftime('%Y-%m-%d %H:00:00', SentimentData.timestamp).label('hour'),
                func.avg(SentimentData.sentiment_score).label('avg_sentiment'),
                func.count(SentimentData.id).label('post_count')
            ).filter(
                SentimentData.timestamp >= cutoff_time
            ).group_by(
                func.strftime('%Y-%m-%d %H:00:00', SentimentData.timestamp)
            ).order_by('hour').all()
            
            trends = []
            for hour_data in hourly_data:
                trends.append({
                    'timestamp': hour_data.hour,
                    'avg_sentiment': float(hour_data.avg_sentiment or 0),
                    'post_count': hour_data.post_count
                })
            
            # Calculate trend direction
            if len(trends) >= 2:
                recent_avg = statistics.mean([t['avg_sentiment'] for t in trends[-6:]])  # Last 6 hours
                earlier_avg = statistics.mean([t['avg_sentiment'] for t in trends[:6]])   # First 6 hours
                trend_direction = 'improving' if recent_avg > earlier_avg else 'declining'
                trend_strength = abs(recent_avg - earlier_avg)
            else:
                trend_direction = 'stable'
                trend_strength = 0.0
            
            self.logger.info(f"Analyzed sentiment trends over {hours} hours: {trend_direction}")
            
            return {
                'hourly_data': trends,
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'total_hours': len(trends)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment trends: {e}")
            return {'hourly_data': [], 'trend_direction': 'unknown', 'trend_strength': 0.0}
        finally:
            session.close()
    
    def get_top_influencers(self, hours=24, limit=10):
        """Get top influencers based on engagement and sentiment impact"""
        recent_data = self.get_recent_sentiment_data(hours)
        
        if not recent_data:
            return []
        
        # Group by author and calculate metrics
        author_metrics = {}
        
        for entry in recent_data:
            author = entry.author
            if author not in author_metrics:
                author_metrics[author] = {
                    'posts': 0,
                    'total_engagement': 0,
                    'sentiment_scores': [],
                    'source': entry.source
                }
            
            author_metrics[author]['posts'] += 1
            author_metrics[author]['total_engagement'] += entry.likes + entry.retweets + entry.replies
            author_metrics[author]['sentiment_scores'].append(entry.sentiment_score)
        
        # Calculate influence score
        influencers = []
        for author, metrics in author_metrics.items():
            if metrics['posts'] < 2:  # Skip authors with only 1 post
                continue
            
            avg_sentiment = statistics.mean(metrics['sentiment_scores'])
            avg_engagement = metrics['total_engagement'] / metrics['posts']
            
            # Influence score based on engagement and consistency
            influence_score = avg_engagement * metrics['posts'] * (abs(avg_sentiment) + 0.1)
            
            influencers.append({
                'author': author,
                'posts': metrics['posts'],
                'avg_engagement': avg_engagement,
                'avg_sentiment': avg_sentiment,
                'influence_score': influence_score,
                'source': metrics['source']
            })
        
        # Sort by influence score and return top N
        influencers.sort(key=lambda x: x['influence_score'], reverse=True)
        return influencers[:limit]
    
    def generate_sentiment_summary(self, hours=24):
        """Generate a comprehensive sentiment summary"""
        overall = self.calculate_overall_sentiment(hours)
        by_source = self.get_sentiment_by_source(hours)
        trends = self.get_sentiment_trends(hours * 3)  # Longer period for trends
        influencers = self.get_top_influencers(hours)
        
        # Generate text summary
        sentiment_text = "positive" if overall['overall_score'] > 0.1 else "negative" if overall['overall_score'] < -0.1 else "neutral"
        confidence_text = "high" if overall['confidence'] > 0.7 else "medium" if overall['confidence'] > 0.4 else "low"
        
        summary_text = f"""
        Bitcoin Sentiment Analysis Summary ({hours}h):
        
        Overall Sentiment: {sentiment_text.title()} ({overall['overall_score']:.3f})
        Confidence Level: {confidence_text.title()} ({overall['confidence']:.2f})
        
        Data Points:
        - Total Posts: {overall['total_posts']}
        - Twitter Posts: {by_source['twitter']['count']}
        - Reddit Posts: {by_source['reddit']['count']}
        
        Distribution:
        - Positive: {overall['positive_ratio']:.1%}
        - Negative: {overall['negative_ratio']:.1%}
        - Neutral: {overall['neutral_ratio']:.1%}
        
        Trend: {trends['trend_direction'].title()} (strength: {trends['trend_strength']:.3f})
        
        Top Influencer: {influencers[0]['author'] if influencers else 'N/A'}
        """
        
        return {
            'summary_text': summary_text.strip(),
            'overall': overall,
            'by_source': by_source,
            'trends': trends,
            'influencers': influencers,
            'timestamp': datetime.utcnow()
        }

if __name__ == "__main__":
    analyzer = SentimentAnalyzer()
    summary = analyzer.generate_sentiment_summary(hours=24)
    print(summary['summary_text'])
