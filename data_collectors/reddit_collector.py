import praw
import logging
from datetime import datetime
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from config.settings import settings
from database.models import get_session, SentimentData
import re

class RedditCollector:
    def __init__(self):
        self.analyzer = SentimentIntensityAnalyzer()
        self.setup_logging()
        self.reddit = self.setup_reddit_client()
        
    def setup_logging(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
    def setup_reddit_client(self):
        """Initialize Reddit API client"""
        try:
            reddit = praw.Reddit(
                client_id=settings.REDDIT_CLIENT_ID,
                client_secret=settings.REDDIT_CLIENT_SECRET,
                user_agent=settings.REDDIT_USER_AGENT
            )
            
            # Test connection
            try:
                reddit.user.me()
                self.logger.info("Reddit API connection successful")
                return reddit
            except Exception:
                # Read-only mode, which is fine for our use case
                self.logger.info("Reddit API in read-only mode")
                return reddit
                
        except Exception as e:
            self.logger.error(f"Failed to setup Reddit client: {e}")
            return None
    
    def clean_text(self, text):
        """Clean Reddit text for analysis"""
        if not text:
            return ""
            
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        # Remove Reddit formatting
        text = re.sub(r'\*\*|__|\*|_|~~|`', '', text)
        # Remove user mentions
        text = re.sub(r'/u/\w+|u/\w+', '', text)
        # Remove subreddit mentions
        text = re.sub(r'/r/\w+|r/\w+', '', text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def analyze_sentiment(self, text):
        """Analyze sentiment using multiple methods"""
        clean_text = self.clean_text(text)
        
        if not clean_text:
            return 0.0, 'neutral'
        
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
    
    def is_bitcoin_related(self, text):
        """Check if text is Bitcoin related"""
        bitcoin_keywords = [
            'bitcoin', 'btc', 'crypto', 'cryptocurrency', 'satoshi',
            'blockchain', 'hodl', 'moon', 'diamond hands', 'bull market',
            'bear market', 'dip', 'ath', 'all time high'
        ]
        
        text_lower = text.lower()
        return any(keyword in text_lower for keyword in bitcoin_keywords)
    
    def collect_posts(self, limit=50):
        """Collect posts from Bitcoin-related subreddits"""
        if not self.reddit:
            self.logger.error("Reddit client not available")
            return []
        
        posts_data = []
        
        for subreddit_name in settings.REDDIT_SUBREDDITS:
            try:
                subreddit = self.reddit.subreddit(subreddit_name)
                
                # Get hot posts
                for post in subreddit.hot(limit=limit):
                    try:
                        # Skip stickied posts
                        if post.stickied:
                            continue
                        
                        # Combine title and selftext
                        full_text = f"{post.title} {post.selftext if post.selftext else ''}"
                        
                        # Check if Bitcoin related (for general crypto subreddits)
                        if subreddit_name in ['CryptoCurrency', 'investing'] and not self.is_bitcoin_related(full_text):
                            continue
                        
                        # Analyze sentiment
                        sentiment_score, sentiment_label = self.analyze_sentiment(full_text)
                        
                        post_data = {
                            'post_id': f"reddit_{post.id}",
                            'text': full_text,
                            'sentiment_score': sentiment_score,
                            'sentiment_label': sentiment_label,
                            'author': str(post.author) if post.author else 'deleted',
                            'likes': post.score,
                            'retweets': 0,  # Reddit doesn't have retweets
                            'replies': post.num_comments,
                            'created_at': datetime.fromtimestamp(post.created_utc),
                            'subreddit': subreddit_name
                        }
                        
                        posts_data.append(post_data)
                        
                    except Exception as e:
                        self.logger.error(f"Error processing post {post.id}: {e}")
                        continue
                        
            except Exception as e:
                self.logger.error(f"Error accessing subreddit {subreddit_name}: {e}")
                continue
        
        self.logger.info(f"Collected {len(posts_data)} Reddit posts")
        return posts_data
    
    def collect_comments(self, limit=20):
        """Collect comments from recent Bitcoin posts"""
        if not self.reddit:
            return []
        
        comments_data = []
        
        try:
            # Get Bitcoin subreddit
            bitcoin_sub = self.reddit.subreddit('Bitcoin')
            
            for post in bitcoin_sub.hot(limit=10):
                try:
                    post.comments.replace_more(limit=5)
                    comment_count = 0
                    
                    for comment in post.comments.list():
                        if comment_count >= limit:
                            break
                            
                        if len(comment.body) < 10:  # Skip very short comments
                            continue
                        
                        # Analyze sentiment
                        sentiment_score, sentiment_label = self.analyze_sentiment(comment.body)
                        
                        comment_data = {
                            'post_id': f"reddit_comment_{comment.id}",
                            'text': comment.body,
                            'sentiment_score': sentiment_score,
                            'sentiment_label': sentiment_label,
                            'author': str(comment.author) if comment.author else 'deleted',
                            'likes': comment.score,
                            'retweets': 0,
                            'replies': len(comment.replies) if hasattr(comment, 'replies') else 0,
                            'created_at': datetime.fromtimestamp(comment.created_utc)
                        }
                        
                        comments_data.append(comment_data)
                        comment_count += 1
                        
                except Exception as e:
                    self.logger.error(f"Error processing comments for post {post.id}: {e}")
                    continue
                    
        except Exception as e:
            self.logger.error(f"Error collecting comments: {e}")
        
        self.logger.info(f"Collected {len(comments_data)} Reddit comments")
        return comments_data
    
    def save_to_database(self, reddit_data):
        """Save Reddit data to database"""
        session = get_session()
        saved_count = 0
        
        try:
            for data in reddit_data:
                # Check if post already exists
                existing = session.query(SentimentData).filter_by(
                    post_id=data['post_id']
                ).first()
                
                if not existing:
                    sentiment_entry = SentimentData(
                        source='reddit',
                        post_id=data['post_id'],
                        text=data['text'],
                        sentiment_score=data['sentiment_score'],
                        sentiment_label=data['sentiment_label'],
                        author=data['author'],
                        likes=data['likes'],
                        retweets=data['retweets'],
                        replies=data['replies'],
                        timestamp=data['created_at']
                    )
                    
                    session.add(sentiment_entry)
                    saved_count += 1
            
            session.commit()
            self.logger.info(f"Saved {saved_count} new Reddit entries to database")
            
        except Exception as e:
            session.rollback()
            self.logger.error(f"Error saving Reddit data to database: {e}")
        finally:
            session.close()
            
        return saved_count
    
    def collect_and_save(self, posts_limit=30, comments_limit=20):
        """Main method to collect Reddit data and save to database"""
        self.logger.info("Starting Reddit data collection...")
        
        # Collect posts
        posts_data = self.collect_posts(limit=posts_limit)
        
        # Collect comments
        comments_data = self.collect_comments(limit=comments_limit)
        
        # Combine all data
        all_data = posts_data + comments_data
        
        if all_data:
            # Save to database
            saved_count = self.save_to_database(all_data)
            
            # Calculate sentiment stats
            positive_count = sum(1 for d in all_data if d['sentiment_label'] == 'positive')
            negative_count = sum(1 for d in all_data if d['sentiment_label'] == 'negative')
            neutral_count = sum(1 for d in all_data if d['sentiment_label'] == 'neutral')
            avg_sentiment = sum(d['sentiment_score'] for d in all_data) / len(all_data)
            
            stats = {
                'total_entries': len(all_data),
                'posts': len(posts_data),
                'comments': len(comments_data),
                'saved_entries': saved_count,
                'positive': positive_count,
                'negative': negative_count,
                'neutral': neutral_count,
                'avg_sentiment': avg_sentiment
            }
            
            self.logger.info(f"Reddit collection complete: {stats}")
            return stats
        else:
            self.logger.warning("No Reddit data collected")
            return None

if __name__ == "__main__":
    collector = RedditCollector()
    collector.collect_and_save(posts_limit=20, comments_limit=15)
