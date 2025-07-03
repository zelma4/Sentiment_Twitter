"""
Advanced Cryptocurrency Sentiment Analysis using CryptoBERT

This module provides enhanced sentiment analysis specifically trained on
cryptocurrency social media content using the CryptoBERT model.
"""

import logging
import numpy as np
from typing import Dict, List, Union

try:
    from transformers import (
        pipeline, 
        AutoTokenizer, 
        AutoModelForSequenceClassification
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CryptoBERTAnalyzer:
    """
    Advanced sentiment analyzer using CryptoBERT model 
    for cryptocurrency content
    """
    
    def __init__(self):
        """Initialize CryptoBERT sentiment analyzer"""
        self.model_name = "ElKulako/cryptobert"
        self.classifier = None
        self._load_model()
    
    def _load_model(self):
        """Load CryptoBERT model with proper meta tensor handling"""
        if not TRANSFORMERS_AVAILABLE:
            logger.error("❌ Transformers library not available")
            raise ImportError("Transformers library required for CryptoBERT")
            
        try:
            logger.info("Loading CryptoBERT model...")
            
            import torch
            
            # Clear any existing cache/memory
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
            
            # Use a simpler approach that avoids meta tensor issues
            try:
                logger.info("Loading CryptoBERT with optimized approach...")
                
                # Load tokenizer first
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    trust_remote_code=True
                )
                
                # Load model with low_cpu_mem_usage to avoid meta tensor issues
                model = AutoModelForSequenceClassification.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float32,
                    low_cpu_mem_usage=True
                )
                
                # Try to move to best available device
                device = "cpu"
                try:
                    if torch.backends.mps.is_available():
                        device = "mps"
                        model = model.to(device)
                        logger.info("Model moved to MPS device")
                    elif torch.cuda.is_available():
                        device = "cuda"
                        model = model.to(device)
                        logger.info("Model moved to CUDA device")
                    else:
                        logger.info("Model staying on CPU")
                except Exception as device_error:
                    logger.warning(f"Device move failed: {device_error}")
                    device = "cpu"
                
                self.classifier = pipeline(
                    "text-classification",
                    model=model,
                    tokenizer=tokenizer,
                    device=0 if device in ["cuda", "mps"] else -1,
                    top_k=None  # Return all scores
                )
                
                logger.info("✅ CryptoBERT model loaded successfully")
                
            except Exception as main_error:
                logger.warning(f"Main loading approach failed: {main_error}")
                # Fallback to simple pipeline loading
                try:
                    tokenizer = AutoTokenizer.from_pretrained(
                        self.model_name,
                        use_fast=True,
                        trust_remote_code=True
                    )
                    
                    model = AutoModelForSequenceClassification.from_pretrained(
                        self.model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float32,
                        low_cpu_mem_usage=True
                    )
                    
                    # Try to move to best available device
                    device = "cpu"
                    if torch.backends.mps.is_available():
                        try:
                            device = "mps"
                            model = model.to(device)
                        except Exception:
                            device = "cpu"
                    elif torch.cuda.is_available():
                        try:
                            device = "cuda"
                            model = model.to(device)
                        except Exception:
                            device = "cpu"
                    
                    self.classifier = pipeline(
                        "text-classification",
                        model=model,
                        tokenizer=tokenizer,
                        device=0 if device in ["cuda", "mps"] else -1,
                        top_k=None
                    )
                    
                    logger.info(f"✅ CryptoBERT loaded on {device} (fallback)")
                    
                except Exception as fallback_error:
                    logger.warning(f"Direct loading failed: {fallback_error}")
                    # Final fallback to simple pipeline
                    self.classifier = pipeline(
                        "text-classification",
                        model=self.model_name,
                        device=-1,  # Force CPU
                        top_k=None
                    )
                    logger.info("✅ CryptoBERT loaded with simple pipeline")
            
        except Exception as e:
            logger.error(f"❌ Failed to load CryptoBERT model: {e}")
            # Complete fallback
            try:
                logger.info("Attempting simple pipeline fallback...")
                self.classifier = pipeline(
                    "text-classification",
                    model=self.model_name,
                    device=-1,
                    top_k=None
                )
                logger.info("✅ CryptoBERT loaded with fallback")
            except Exception as e2:
                logger.error(f"❌ Complete failure to load CryptoBERT: {e2}")
                self.classifier = None
                # Don't raise - allow graceful degradation
    
    def analyze_sentiment(
        self, text: str
    ) -> Dict[str, Union[float, str, int]]:
        """
        Analyze sentiment of a single text using CryptoBERT
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with sentiment score, confidence, label, and raw scores
        """
        if not text or not text.strip():
            return {
                'score': 0.0,
                'confidence': 0.0,
                'label': 'Neutral',
                'raw_scores': {
                    'Bearish': 0.33, 'Neutral': 0.34, 'Bullish': 0.33
                }
            }
        
        # Check if model loaded successfully
        if self.classifier is None:
            logger.warning("CryptoBERT not available, returning neutral sentiment")
            return {
                'score': 0.0,
                'confidence': 0.5,
                'label': 'Neutral',
                'raw_scores': {'Bearish': 0.33, 'Neutral': 0.34, 'Bullish': 0.33}
            }
        
        try:
            # Clean and preprocess text
            clean_text = self._preprocess_text(text)
            
            # Get predictions for all labels
            results = self.classifier(clean_text)
            
            # CryptoBERT returns all scores, find the highest confidence
            if isinstance(results, list) and len(results) > 0:
                if isinstance(results[0], list):
                    # All scores returned
                    scores = results[0]
                else:
                    # Single best result
                    scores = results
            else:
                scores = results
            
            # Convert to our format
            sentiment_map = {
                'Bearish': -1,
                'Neutral': 0, 
                'Bullish': 1
            }
            
            # Find best prediction
            best_prediction = max(scores, key=lambda x: x['score'])
            sentiment_score = sentiment_map.get(best_prediction['label'], 0)
            
            # Create raw scores dict
            raw_scores = {item['label']: item['score'] for item in scores}
            
            return {
                'score': sentiment_score,
                'confidence': best_prediction['score'],
                'label': best_prediction['label'],
                'raw_scores': raw_scores
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {e}")
            return {
                'score': 0.0,
                'confidence': 0.0,
                'label': 'Neutral',
                'raw_scores': {'Bearish': 0.33, 'Neutral': 0.34, 'Bullish': 0.33}
            }
    
    def analyze_batch(self, texts: List[str]) -> List[Dict[str, Union[float, str]]]:
        """
        Analyze sentiment for multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            List of sentiment analysis results
        """
        results = []
        for text in texts:
            results.append(self.analyze_sentiment(text))
        return results
    
    def get_aggregated_sentiment(self, texts: List[str]) -> Dict[str, float]:
        """
        Get aggregated sentiment scores from multiple texts
        
        Args:
            texts: List of texts to analyze
            
        Returns:
            Dict with aggregated sentiment metrics
        """
        if not texts:
            return {
                'mean_sentiment': 0.0,
                'sentiment_std': 0.0,
                'bullish_ratio': 0.0,
                'bearish_ratio': 0.0,
                'neutral_ratio': 0.0,
                'confidence_avg': 0.0
            }
        
        results = self.analyze_batch(texts)
        
        # Extract metrics
        scores = [r['score'] for r in results]
        confidences = [r['confidence'] for r in results]
        labels = [r['label'] for r in results]
        
        # Calculate ratios
        total_count = len(labels)
        bullish_count = labels.count('Bullish')
        bearish_count = labels.count('Bearish')
        neutral_count = labels.count('Neutral')
        
        return {
            'mean_sentiment': np.mean(scores),
            'sentiment_std': np.std(scores),
            'bullish_ratio': bullish_count / total_count,
            'bearish_ratio': bearish_count / total_count,
            'neutral_ratio': neutral_count / total_count,
            'confidence_avg': np.mean(confidences),
            'total_analyzed': total_count
        }
    
    def _preprocess_text(self, text: str) -> str:
        """
        Preprocess text for better CryptoBERT analysis
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        # Basic cleaning
        text = text.strip()
        
        # Remove excessive whitespace
        text = ' '.join(text.split())
        
        # Limit length to CryptoBERT optimal range
        if len(text) > 500:  # Conservative limit
            text = text[:500]
        
        return text


class HybridSentimentAnalyzer:
    """
    Hybrid sentiment analyzer combining CryptoBERT with traditional methods
    """
    
    def __init__(self):
        """Initialize hybrid analyzer"""
        self.cryptobert = CryptoBERTAnalyzer()
        
        # Import traditional analyzers as fallback
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
            self.vader = SentimentIntensityAnalyzer()
        except ImportError:
            logger.warning("VADER not available")
            self.vader = None
            
        try:
            from textblob import TextBlob
            self.textblob_available = True
        except ImportError:
            logger.warning("TextBlob not available")
            self.textblob_available = False
    
    def analyze_comprehensive(self, text: str) -> Dict[str, Union[float, str]]:
        """
        Comprehensive sentiment analysis using multiple methods
        
        Args:
            text: Text to analyze
            
        Returns:
            Dict with comprehensive sentiment analysis
        """
        # Get CryptoBERT analysis (primary)
        cryptobert_result = self.cryptobert.analyze_sentiment(text)
        
        # Get traditional sentiment scores for comparison
        vader_score = 0.0
        textblob_score = 0.0
        
        if self.vader:
            try:
                vader_result = self.vader.polarity_scores(text)
                vader_score = vader_result['compound']
            except:
                pass
        
        if self.textblob_available:
            try:
                from textblob import TextBlob
                blob = TextBlob(text)
                textblob_score = blob.sentiment.polarity
            except:
                pass
        
        # Calculate hybrid score (CryptoBERT weighted more heavily)
        cryptobert_score = cryptobert_result['score']
        
        # Weighted combination (CryptoBERT 70%, VADER 20%, TextBlob 10%)
        hybrid_score = (
            0.7 * cryptobert_score +
            0.2 * vader_score +
            0.1 * textblob_score
        )
        
        return {
            'primary_score': cryptobert_score,
            'primary_label': cryptobert_result['label'],
            'primary_confidence': cryptobert_result['confidence'],
            'hybrid_score': hybrid_score,
            'vader_score': vader_score,
            'textblob_score': textblob_score,
            'raw_cryptobert': cryptobert_result['raw_scores']
        }


def test_cryptobert():
    """Test function for CryptoBERT analyzer"""
    analyzer = CryptoBERTAnalyzer()
    
    test_texts = [
        "Bitcoin is going to the moon! 🚀 HODL forever!",
        "Crypto market is crashing, everything is down badly",
        "Bitcoin price is stable today, nothing much happening",
        "Just bought more BTC, feeling bullish about the future!",
        "Sold all my crypto, this bear market is brutal"
    ]
    
    print("🧠 Testing CryptoBERT Sentiment Analysis:")
    print("=" * 60)
    
    for text in test_texts:
        result = analyzer.analyze_sentiment(text)
        print(f"Text: {text[:50]}...")
        print(f"Label: {result['label']} | Score: {result['score']:.2f} | Confidence: {result['confidence']:.3f}")
        print(f"Raw scores: {result['raw_scores']}")
        print("-" * 60)
    
    # Test aggregated analysis
    print("\n📊 Aggregated Analysis:")
    agg_result = analyzer.get_aggregated_sentiment(test_texts)
    for key, value in agg_result.items():
        print(f"{key}: {value:.3f}")


if __name__ == "__main__":
    test_cryptobert()
