"""
Advanced Sentiment Analysis using RoBERTa and FinBERT
Research shows RoBERTa/FinBERT gives ~2.01% MAPE vs basic VADER
"""

import logging
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Union

# RoBERTa and FinBERT imports
try:
    from transformers import (
        RobertaTokenizer, RobertaForSequenceClassification,
        pipeline, AutoTokenizer, AutoModelForSequenceClassification
    )
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# Fallback to VADER if needed
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    VADER_AVAILABLE = False

class AdvancedSentimentAnalyzer:
    """
    Advanced sentiment analyzer using RoBERTa and FinBERT
    Falls back to VADER if transformers not available
    """
    
    def __init__(self, use_finbert=True, use_roberta=True):
        self.setup_logging()
        
        self.use_finbert = use_finbert
        self.use_roberta = use_roberta
        
        # Initialize models
        self.finbert_analyzer = None
        self.roberta_analyzer = None
        self.vader_analyzer = None
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize analyzers
        self._initialize_analyzers()
        
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
    
    def _initialize_analyzers(self):
        """Initialize all available sentiment analyzers"""
        if not TRANSFORMERS_AVAILABLE:
            self.logger.warning("🚨 Transformers not available, falling back to VADER")
            self._initialize_vader()
            return
        
        # Initialize FinBERT for financial sentiment
        if self.use_finbert:
            try:
                self.logger.info("📊 Loading FinBERT for financial sentiment...")
                self.finbert_analyzer = pipeline(
                    "sentiment-analysis",
                    model="ProsusAI/finbert",
                    tokenizer="ProsusAI/finbert",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.logger.info("✅ FinBERT loaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to load FinBERT: {e}")
                self.finbert_analyzer = None
        
        # Initialize RoBERTa for general sentiment
        if self.use_roberta:
            try:
                self.logger.info("🤖 Loading RoBERTa for general sentiment...")
                self.roberta_analyzer = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                    device=0 if torch.cuda.is_available() else -1
                )
                self.logger.info("✅ RoBERTa loaded successfully")
            except Exception as e:
                self.logger.error(f"❌ Failed to load RoBERTa: {e}")
                self.roberta_analyzer = None
        
        # Initialize VADER as fallback
        if self.finbert_analyzer is None and self.roberta_analyzer is None:
            self.logger.warning("⚠️ No transformer models available, using VADER")
            self._initialize_vader()
    
    def _initialize_vader(self):
        """Initialize VADER as fallback"""
        if VADER_AVAILABLE:
            self.vader_analyzer = SentimentIntensityAnalyzer()
            self.logger.info("✅ VADER sentiment analyzer initialized")
        else:
            self.logger.error("❌ No sentiment analyzers available!")
    
    def analyze_text(self, text: str, context: str = "general") -> Dict:
        """
        Analyze sentiment of text using best available model
        
        Args:
            text: Text to analyze
            context: Context type ("financial", "general", "crypto")
            
        Returns:
            Dictionary with sentiment analysis results
        """
        try:
            # Choose best model based on context
            if context in ["financial", "crypto"] and self.finbert_analyzer:
                return self._analyze_with_finbert(text)
            elif self.roberta_analyzer:
                return self._analyze_with_roberta(text)
            elif self.vader_analyzer:
                return self._analyze_with_vader(text)
            else:
                return self._get_default_sentiment()
                
        except Exception as e:
            self.logger.error(f"Error analyzing text: {e}")
            return self._get_default_sentiment()
    
    def _analyze_with_finbert(self, text: str) -> Dict:
        """Analyze with FinBERT"""
        try:
            # Truncate text if too long
            text = text[:512]
            
            result = self.finbert_analyzer(text)
            
            # FinBERT returns positive/negative/neutral
            label = result[0]['label'].lower()
            confidence = result[0]['score']
            
            # Convert to standardized format
            if label == 'positive':
                sentiment_score = confidence
                sentiment_label = 'positive'
            elif label == 'negative':
                sentiment_score = -confidence
                sentiment_label = 'negative'
            else:  # neutral
                sentiment_score = 0.0
                sentiment_label = 'neutral'
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'model': 'finbert',
                'raw_result': result[0],
                'compound': sentiment_score,  # For compatibility
                'positive': confidence if label == 'positive' else 0.0,
                'negative': confidence if label == 'negative' else 0.0,
                'neutral': confidence if label == 'neutral' else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"FinBERT analysis error: {e}")
            return self._get_default_sentiment()
    
    def _analyze_with_roberta(self, text: str) -> Dict:
        """Analyze with RoBERTa"""
        try:
            # Truncate text if too long
            text = text[:512]
            
            result = self.roberta_analyzer(text)
            
            # RoBERTa returns LABEL_0 (negative), LABEL_1 (neutral), LABEL_2 (positive)
            label = result[0]['label']
            confidence = result[0]['score']
            
            # Convert to standardized format
            if label == 'LABEL_2':  # positive
                sentiment_score = confidence
                sentiment_label = 'positive'
            elif label == 'LABEL_0':  # negative
                sentiment_score = -confidence
                sentiment_label = 'negative'
            else:  # LABEL_1 - neutral
                sentiment_score = 0.0
                sentiment_label = 'neutral'
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment_label': sentiment_label,
                'confidence': confidence,
                'model': 'roberta',
                'raw_result': result[0],
                'compound': sentiment_score,  # For compatibility
                'positive': confidence if label == 'LABEL_2' else 0.0,
                'negative': confidence if label == 'LABEL_0' else 0.0,
                'neutral': confidence if label == 'LABEL_1' else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"RoBERTa analysis error: {e}")
            return self._get_default_sentiment()
    
    def _analyze_with_vader(self, text: str) -> Dict:
        """Analyze with VADER (fallback)"""
        try:
            scores = self.vader_analyzer.polarity_scores(text)
            
            # Determine label
            compound = scores['compound']
            if compound >= 0.05:
                sentiment_label = 'positive'
            elif compound <= -0.05:
                sentiment_label = 'negative'
            else:
                sentiment_label = 'neutral'
            
            return {
                'sentiment_score': compound,
                'sentiment_label': sentiment_label,
                'confidence': abs(compound),
                'model': 'vader',
                'raw_result': scores,
                'compound': compound,
                'positive': scores['pos'],
                'negative': scores['neg'],
                'neutral': scores['neu']
            }
            
        except Exception as e:
            self.logger.error(f"VADER analysis error: {e}")
            return self._get_default_sentiment()
    
    def _get_default_sentiment(self) -> Dict:
        """Return default neutral sentiment"""
        return {
            'sentiment_score': 0.0,
            'sentiment_label': 'neutral',
            'confidence': 0.0,
            'model': 'default',
            'raw_result': {},
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 1.0
        }
    
    def analyze_batch(self, texts: List[str], context: str = "general") -> List[Dict]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            context: Context type
            
        Returns:
            List of sentiment analysis results
        """
        try:
            results = []
            
            for text in texts:
                result = self.analyze_text(text, context)
                results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Batch analysis error: {e}")
            return [self._get_default_sentiment() for _ in texts]
    
    def get_sentiment_momentum(self, texts: List[str], context: str = "general") -> Dict:
        """
        Calculate sentiment momentum from a series of texts
        Key feature from research showing high importance
        
        Args:
            texts: List of texts in chronological order
            context: Context type
            
        Returns:
            Dictionary with momentum metrics
        """
        try:
            # Analyze all texts
            sentiments = self.analyze_batch(texts, context)
            
            # Extract sentiment scores
            scores = [s['sentiment_score'] for s in sentiments]
            
            if len(scores) < 2:
                return {'momentum': 0.0, 'trend': 'neutral', 'volatility': 0.0}
            
            # Calculate momentum (rate of change)
            momentum = np.diff(scores)
            avg_momentum = np.mean(momentum)
            
            # Calculate volatility
            volatility = np.std(scores)
            
            # Determine trend
            if avg_momentum > 0.1:
                trend = 'bullish'
            elif avg_momentum < -0.1:
                trend = 'bearish'
            else:
                trend = 'neutral'
            
            return {
                'momentum': float(avg_momentum),
                'trend': trend,
                'volatility': float(volatility),
                'scores': scores,
                'latest_score': scores[-1],
                'score_change': scores[-1] - scores[0] if len(scores) > 1 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Sentiment momentum calculation error: {e}")
            return {'momentum': 0.0, 'trend': 'neutral', 'volatility': 0.0}
    
    def get_enhanced_features(self, texts: List[str], context: str = "crypto") -> Dict:
        """
        Extract enhanced sentiment features for ML models
        Based on research showing importance of compound score, negativity, etc.
        
        Args:
            texts: List of texts to analyze
            context: Context type
            
        Returns:
            Dictionary with enhanced features
        """
        try:
            # Analyze all texts
            sentiments = self.analyze_batch(texts, context)
            
            # Extract basic metrics
            scores = [s['sentiment_score'] for s in sentiments]
            compounds = [s['compound'] for s in sentiments]
            negatives = [s['negative'] for s in sentiments]
            positives = [s['positive'] for s in sentiments]
            
            # Calculate momentum
            momentum_data = self.get_sentiment_momentum(texts, context)
            
            # Enhanced features
            features = {
                # Basic sentiment metrics
                'sentiment_mean': np.mean(scores),
                'sentiment_std': np.std(scores),
                'sentiment_latest': scores[-1] if scores else 0.0,
                
                # Compound score (important from research)
                'compound_mean': np.mean(compounds),
                'compound_std': np.std(compounds),
                'compound_latest': compounds[-1] if compounds else 0.0,
                
                # Negativity (important from research)
                'negativity_mean': np.mean(negatives),
                'negativity_std': np.std(negatives),
                'negativity_latest': negatives[-1] if negatives else 0.0,
                
                # Positivity
                'positivity_mean': np.mean(positives),
                'positivity_std': np.std(positives),
                'positivity_latest': positives[-1] if positives else 0.0,
                
                # Momentum features (key from research)
                'sentiment_momentum': momentum_data['momentum'],
                'sentiment_trend': momentum_data['trend'],
                'sentiment_volatility': momentum_data['volatility'],
                'sentiment_change': momentum_data['score_change'],
                
                # Distribution features
                'sentiment_skew': float(pd.Series(scores).skew()) if len(scores) > 2 else 0.0,
                'sentiment_kurtosis': float(pd.Series(scores).kurtosis()) if len(scores) > 2 else 0.0,
                
                # Ratio features
                'pos_neg_ratio': np.mean(positives) / (np.mean(negatives) + 1e-8),
                'extreme_sentiment_ratio': len([s for s in scores if abs(s) > 0.5]) / len(scores) if scores else 0.0,
                
                # Time-based features
                'sentiment_persistence': self._calculate_persistence(scores),
                'sentiment_reversals': self._count_reversals(scores),
                
                # Model information
                'model_used': sentiments[0]['model'] if sentiments else 'default',
                'analysis_timestamp': datetime.utcnow().isoformat()
            }
            
            return features
            
        except Exception as e:
            self.logger.error(f"Enhanced features extraction error: {e}")
            return self._get_default_features()
    
    def _calculate_persistence(self, scores: List[float]) -> float:
        """Calculate sentiment persistence (how long sentiment stays in same direction)"""
        if len(scores) < 2:
            return 0.0
        
        try:
            # Count periods where sentiment doesn't change direction
            same_direction_count = 0
            for i in range(1, len(scores)):
                if (scores[i] > 0) == (scores[i-1] > 0):
                    same_direction_count += 1
            
            return same_direction_count / (len(scores) - 1)
        except:
            return 0.0
    
    def _count_reversals(self, scores: List[float]) -> int:
        """Count sentiment reversals (positive to negative or vice versa)"""
        if len(scores) < 2:
            return 0
        
        try:
            reversals = 0
            for i in range(1, len(scores)):
                if (scores[i] > 0) != (scores[i-1] > 0):
                    reversals += 1
            
            return reversals
        except:
            return 0
    
    def _get_default_features(self) -> Dict:
        """Return default features when analysis fails"""
        return {
            'sentiment_mean': 0.0,
            'sentiment_std': 0.0,
            'sentiment_latest': 0.0,
            'compound_mean': 0.0,
            'compound_std': 0.0,
            'compound_latest': 0.0,
            'negativity_mean': 0.0,
            'negativity_std': 0.0,
            'negativity_latest': 0.0,
            'positivity_mean': 0.0,
            'positivity_std': 0.0,
            'positivity_latest': 0.0,
            'sentiment_momentum': 0.0,
            'sentiment_trend': 'neutral',
            'sentiment_volatility': 0.0,
            'sentiment_change': 0.0,
            'sentiment_skew': 0.0,
            'sentiment_kurtosis': 0.0,
            'pos_neg_ratio': 1.0,
            'extreme_sentiment_ratio': 0.0,
            'sentiment_persistence': 0.0,
            'sentiment_reversals': 0,
            'model_used': 'default',
            'analysis_timestamp': datetime.utcnow().isoformat()
        }
    
    def get_model_info(self) -> Dict:
        """Get information about available models"""
        return {
            'finbert_available': self.finbert_analyzer is not None,
            'roberta_available': self.roberta_analyzer is not None,
            'vader_available': self.vader_analyzer is not None,
            'transformers_available': TRANSFORMERS_AVAILABLE,
            'device': str(self.device),
            'recommended_context': {
                'financial': 'finbert' if self.finbert_analyzer else 'roberta',
                'crypto': 'finbert' if self.finbert_analyzer else 'roberta',
                'general': 'roberta' if self.roberta_analyzer else 'vader'
            }
        }
    
    def analyze_sentiment(self, text: str, context: str = "general") -> Dict:
        """
        Main sentiment analysis method (alias for analyze_text)
        
        Args:
            text: Text to analyze
            context: Context type ("financial", "general", "crypto")
            
        Returns:
            Dictionary with sentiment analysis results
        """
        return self.analyze_text(text, context)

    # Database interaction methods for compatibility
    def get_recent_sentiment_data(self, hours=24):
        """Get recent sentiment data from database"""
        from database.models import get_session, SentimentData
        from datetime import timedelta
        
        session = get_session()
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_data = session.query(SentimentData).filter(
                SentimentData.timestamp >= cutoff_time
            ).order_by(SentimentData.timestamp).all()
            return recent_data
        except Exception as e:
            self.logger.error(f"Error fetching recent sentiment data: {e}")
            return []
        finally:
            session.close()
    
    def get_recent_sentiment_dataframe(self, hours=24):
        """Get sentiment data as DataFrame for ML models"""
        recent_data = self.get_recent_sentiment_data(hours)
        
        if not recent_data:
            return pd.DataFrame()
        
        data = []
        for entry in recent_data:
            try:
                confidence = getattr(entry, 'confidence', abs(entry.sentiment_score))
                text = getattr(entry, 'text', '')
                source = getattr(entry, 'source_platform', getattr(entry, 'source', 'unknown'))
                
                data.append({
                    'date': entry.timestamp.date(),
                    'sentiment': entry.sentiment_score,
                    'confidence': confidence,
                    'text': text,
                    'source': source
                })
            except Exception as e:
                self.logger.warning(f"Skipping sentiment entry due to error: {e}")
                continue
        
        if not data:
            return pd.DataFrame()
        
        df = pd.DataFrame(data)
        daily_sentiment = df.groupby('date').agg({
            'sentiment': 'mean',
            'confidence': 'mean'
        }).reset_index()
        
        return daily_sentiment
    
    def generate_sentiment_summary(self, hours=24):
        """Generate a comprehensive sentiment summary"""
        recent_data = self.get_recent_sentiment_data(hours)
        
        if not recent_data:
            return {
                'summary_text': 'No sentiment data available',
                'overall': {'overall_score': 0.0, 'confidence': 0.0, 'total_posts': 0},
                'by_source': {},
                'trends': {'trend_direction': 'neutral', 'trend_strength': 0.0},
                'timestamp': datetime.utcnow()
            }
        
        # Calculate overall metrics
        scores = [entry.sentiment_score for entry in recent_data]
        total_posts = len(recent_data)
        positive_count = sum(1 for score in scores if score > 0.05)
        negative_count = sum(1 for score in scores if score < -0.05)
        
        overall_score = sum(scores) / total_posts if total_posts > 0 else 0.0
        overall_confidence = sum(abs(score) for score in scores) / total_posts if total_posts > 0 else 0.0
        
        # Source breakdown
        sources = {}
        for entry in recent_data:
            source = getattr(entry, 'source_platform', 'unknown')
            if source not in sources:
                sources[source] = []
            sources[source].append(entry.sentiment_score)
        
        by_source = {}
        for source, scores_list in sources.items():
            by_source[source] = {
                'average_sentiment': sum(scores_list) / len(scores_list),
                'count': len(scores_list)
            }
        
        # Simple trend
        if len(scores) >= 10:
            early_avg = sum(scores[:len(scores)//3]) / (len(scores)//3)
            late_avg = sum(scores[-len(scores)//3:]) / (len(scores)//3)
            change = late_avg - early_avg
            trend_direction = 'bullish' if change > 0.1 else 'bearish' if change < -0.1 else 'neutral'
            trend_strength = abs(change)
        else:
            trend_direction = 'neutral'
            trend_strength = 0.0
        
        # Text summary
        sentiment_text = "positive" if overall_score > 0.1 else "negative" if overall_score < -0.1 else "neutral"
        confidence_text = "high" if overall_confidence > 0.7 else "medium" if overall_confidence > 0.4 else "low"
        
        twitter_count = by_source.get('twitter', {}).get('count', 0)
        reddit_count = by_source.get('reddit', {}).get('count', 0)
        
        summary_text = f"""Bitcoin Sentiment Analysis Summary ({hours}h):
        
Overall Sentiment: {sentiment_text.title()} ({overall_score:.3f})
Confidence Level: {confidence_text.title()} ({overall_confidence:.2f})

Data Points:
- Total Posts: {total_posts}
- Twitter Posts: {twitter_count}
- Reddit Posts: {reddit_count}

Distribution:
- Positive: {positive_count/total_posts:.1%}
- Negative: {negative_count/total_posts:.1%}
- Neutral: {(total_posts-positive_count-negative_count)/total_posts:.1%}

Trend: {trend_direction.title()} (strength: {trend_strength:.3f})"""
        
        return {
            'summary_text': summary_text.strip(),
            'overall': {
                'overall_score': overall_score,
                'confidence': overall_confidence,
                'total_posts': total_posts,
                'positive_ratio': positive_count/total_posts,
                'negative_ratio': negative_count/total_posts,
                'neutral_ratio': (total_posts-positive_count-negative_count)/total_posts
            },
            'by_source': by_source,
            'trends': {
                'trend_direction': trend_direction,
                'trend_strength': trend_strength,
                'change_24h': change if 'change' in locals() else 0.0
            },
            'timestamp': datetime.utcnow()
        }
