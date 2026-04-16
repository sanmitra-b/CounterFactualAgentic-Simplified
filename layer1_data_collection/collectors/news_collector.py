"""
News Collector Module
Fetches news from NewsAPI and RSS feeds.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import List, Dict
import requests
import feedparser

logger = logging.getLogger(__name__)


class NewsCollector:
    """Collects financial news from multiple free sources."""
    
    def __init__(self, config: Dict):
        """
        Initialize the news collector with configuration.
        
        Args:
            config: Dictionary containing 'news' settings from config.json
        """
        self.config = config
        self.articles = []
        
    def collect_from_newsapi(self) -> List[Dict]:
        """
        Fetch articles from NewsAPI free tier.
        
        Returns:
            List of article dictionaries with normalized fields.
        """
        if not self.config.get("methods", {}).get("newsapi", {}).get("enabled"):
            logger.info("NewsAPI disabled in config")
            return []
        
        api_key = self.config["methods"]["newsapi"].get("api_key")
        if api_key == "YOUR_NEWSAPI_KEY_HERE":
            logger.warning("NewsAPI key not configured. Skipping NewsAPI.")
            return []
        
        base_url = "https://newsapi.org/v2/everything"
        articles = []
        
        # Search for each keyword
        for keyword in self.config.get("keywords", []):
            try:
                # Get articles from last 30 days
                from_date = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
                
                params = {
                    "q": keyword,
                    "from": from_date,
                    "sortBy": "publishedAt",
                    "language": "en",
                    "apiKey": api_key,
                    "pageSize": 20  # Max per request (100 per day limit)
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                if data.get("status") == "ok":
                    for article in data.get("articles", []):
                        articles.append({
                            "title": article.get("title", ""),
                            "source": article.get("source", {}).get("name", "Unknown"),
                            "date": article.get("publishedAt", ""),
                            "url": article.get("url", ""),
                            "content": article.get("content") or article.get("description", ""),
                            "author": article.get("author", "Unknown")
                        })
                    logger.info(f"NewsAPI: Found {len(data.get('articles', []))} articles for '{keyword}'")
                else:
                    logger.warning(f"NewsAPI error: {data.get('message')}")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"NewsAPI request failed for '{keyword}': {e}")
                continue
        
        return articles
    
    def collect_from_rss(self) -> List[Dict]:
        """
        Fetch articles from RSS feeds.
        
        Returns:
            List of article dictionaries with normalized fields.
        """
        if not self.config.get("methods", {}).get("rss", {}).get("enabled"):
            logger.info("RSS disabled in config")
            return []
        
        feeds = self.config["methods"]["rss"].get("feeds", [])
        articles = []
        
        for feed_url in feeds:
            try:
                feed = feedparser.parse(feed_url)
                
                # Limit entries per feed
                max_entries = self.config.get("max_results_per_source", 50)
                
                for entry in feed.entries[:max_entries]:
                    # Extract date if available
                    if hasattr(entry, 'published'):
                        date = entry.published
                    elif hasattr(entry, 'updated'):
                        date = entry.updated
                    else:
                        date = datetime.now().isoformat()
                    
                    articles.append({
                        "title": entry.get("title", ""),
                        "source": feed.feed.get("title", "RSS Feed"),
                        "date": date,
                        "url": entry.get("link", ""),
                        "content": entry.get("summary", ""),
                        "author": entry.get("author", "Unknown")
                    })
                
                logger.info(f"RSS: Found {len(feed.entries[:max_entries])} articles from {feed.feed.get('title', 'Unknown')}")
                
            except Exception as e:
                logger.error(f"RSS parsing failed for {feed_url}: {e}")
                continue
        
        return articles
    
    def collect(self) -> List[Dict]:
        """
        Collect news from all enabled sources.
        
        Returns:
            List of all collected articles.
        """
        logger.info("Starting news collection...")
        
        all_articles = []
        
        # Collect from NewsAPI
        all_articles.extend(self.collect_from_newsapi())
        
        # Collect from RSS feeds
        all_articles.extend(self.collect_from_rss())
        
        self.articles = all_articles
        logger.info(f"News collection complete. Total articles: {len(self.articles)}")
        
        return self.articles


def collect_news(config: Dict) -> List[Dict]:
    """
    Convenience function to collect news.
    
    Args:
        config: Dictionary with news configuration.
    
    Returns:
        List of article dictionaries.
    """
    collector = NewsCollector(config)
    return collector.collect()
