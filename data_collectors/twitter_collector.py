import tweepy
import time
import logging
from datetime import datetime, timedelta
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from config.settings import settings
from database.models import get_session, SentimentData
import re

class TwitterCollector:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.setup_logging()
        self.client = self.setup_twitter_client()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_twitter_client(self):
        """Initialize Twitter API client"""
        try:
            # Using Twitter API v2
            client = tweepy.Client(
                bearer_token=settings.TWITTER_BEARER_TOKEN,
                consumer_key=settings.TWITTER_API_KEY,
                consumer_secret=settings.TWITTER_API_SECRET,
                access_token=settings.TWITTER_ACCESS_TOKEN,
                access_token_secret=settings.TWITTER_ACCESS_TOKEN_SECRET,
                wait_on_rate_limit=True
            )
            
            # Test connection
            try:
                client.get_me()
                self.logger.info("Twitter API connection successful")
                return client
            except Exception as e:
                self.logger.error(f"Twitter API test failed: {e}")
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to setup Twitter client: {e}")
            return None
    
    def clean_tweet(self, text):
        """Clean tweet text for analysis"""
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove user mentions and hashtags for cleaner analysis
        text = re.sub(r'@\w+|#\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using multiple methods"""
        clean_text = self.clean_tweet(text)
        
        # VADER sentiment
        vader_scores = self.analyzer.polarity_scores(clean_text)
        vader_compound = vader_scores['compound']
        
        # TextBlob sentiment
        blob = TextBlob(clean_text)
        textblob_polarity = blob.sentiment.polarity
        
        # Average the two methods
        avg_sentiment = (vader_compound + textblob_polarity) / 2
        
        # Classify sentiment
        if avg_sentiment >= 0.1:
            label = 'positive'
        elif avg_sentiment <= -0.1:
            label = 'negative'
        else:
            label = 'neutral'
            
        return avg_sentiment, label
    
    def search_tweets(self, max_results=100):
        """Search for Bitcoin-related tweets"""
        if not self.client:
            self.logger.error("Twitter client not available")
            return []
        
        tweets_data = []
        
        try:
            # Build search query
            query = ' OR '.join(settings.TWITTER_KEYWORDS)
            query += ' -is:retweet lang:en'  # Exclude retweets, English only
            
            # Search tweets
            tweets = tweepy.Paginator(
                self.client.search_recent_tweets,
                query=query,
                tweet_fields=['created_at', 'author_id', 'public_metrics', 'context_annotations'],
                max_results=min(max_results, 100),  # API limit
                limit=1
            ).flatten(limit=max_results)
            
            for tweet in tweets:
                try:
                    # Analyze sentiment
                    sentiment_score, sentiment_label = self.analyze_sentiment(tweet.text)
                    
                    # Get metrics
                    metrics = tweet.public_metrics
                    
                    tweet_data = {
                        'post_id': str(tweet.id),
                        'text': tweet.text,
                        'sentiment_score': sentiment_score,
                        'sentiment_label': sentiment_label,
                        'author': str(tweet.author_id),
                        'likes': metrics.get('like_count', 0),
                        'retweets': metrics.get('retweet_count', 0),
                        'replies': metrics.get('reply_count', 0),
                        'created_at': tweet.created_at
                    }
                    
                    tweets_data.append(tweet_data)
                    
                except Exception as e:
                    self.logger.error(f"Error processing tweet {tweet.id}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error searching tweets: {e}")
            
        self.logger.info(f"Collected {len(tweets_data)} tweets")
        return tweets_data
    
    def save_to_database(self, tweets_data):
        """Save tweets to database"""
        session = get_session()
        saved_count = 0
        
        try:
            for tweet_data in tweets_data:
                # Check if tweet already exists
                existing = session.query(SentimentData).filter_by(
                    post_id=tweet_data['post_id']
                ).first()
                
                if not existing:
                    sentiment_entry = SentimentData(
                        source='twitter',
                        post_id=tweet_data['post_id'],
                        text=tweet_data['text'],
                        sentiment_score=tweet_data['sentiment_score'],
                        sentiment_label=tweet_data['sentiment_label'],
                        author=tweet_data['author'],
                        likes=tweet_data['likes'],
                        retweets=tweet_data['retweets'],
                        replies=tweet_data['replies'],
                        timestamp=tweet_data.get('created_at', datetime.utcnow())
                    )
                    
                    session.add(sentiment_entry)
                    saved_count += 1
            
            session.commit()
            self.logger.info(f"Saved {saved_count} new tweets to database")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving tweets to database: {e}")
        finally:
            session.close()
            
        return saved_count
    
    def collect_and_save(self, max_results=100):
        """Main method to collect tweets and save to database"""
        self.logger.info("Starting Twitter data collection...")
        
        # Collect tweets
        tweets_data = self.search_tweets(max_results)
        
        if tweets_data:
            # Save to database
            saved_count = self.save_to_database(tweets_data)
            
            # Calculate sentiment stats
            positive_count = sum(1 for t in tweets_data if t['sentiment_label'] == 'positive')
            negative_count = sum(1 for t in tweets_data if t['sentiment_label'] == 'negative')
            neutral_count = sum(1 for t in tweets_data if t['sentiment_label'] == 'neutral')
            avg_sentiment = sum(t['sentiment_score'] for t in tweets_data) / len(tweets_data)
            
            stats = {
                'total_tweets': len(tweets_data),
                'saved_tweets': saved_count,
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count,
                'avg_sentiment': avg_sentiment
            }
            
            self.logger.info(f"Collection complete: {stats}")
            return stats
        else:
            self.logger.warning("No tweets collected")
            return None

if __name__ == "__main__":
    collector = TwitterCollector()
    collector.collect_and_save(max_results=50)
