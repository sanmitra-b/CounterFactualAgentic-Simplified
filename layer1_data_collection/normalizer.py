"""
Data Normalizer Module
Converts all data from different sources to a common schema.
"""

import logging
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class DataNormalizer:
    """Normalizes data from different sources into a unified schema."""
    
    # Common schema for all data
    SCHEMA = {
        "source_type": "news|financial|jobs|social",
        "source_name": "NewsAPI|RSS|Alpha Vantage|FRED|Adzuna|USAJOBS|Pushshift",
        "date": "YYYY-MM-DD or ISO format",
        "content": "main text or primary value",
        "metadata": {
            "description": "source-specific additional fields"
        }
    }
    
    @staticmethod
    def _parse_date(date_str: str) -> str:
        """
        Parse various date formats to ISO format (YYYY-MM-DD).
        
        Args:
            date_str: Date string in various formats.
        
        Returns:
            Normalized date string in YYYY-MM-DD format.
        """
        if not date_str:
            return datetime.now().strftime("%Y-%m-%d")
        
        # Try common formats
        formats = [
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d",
            "%a, %d %b %Y %H:%M:%S %z",
            "%a, %d %b %Y %H:%M:%S GMT"
        ]
        
        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.replace("Z", "+0000"), fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        
        # If parsing fails, return today's date
        logger.warning(f"Could not parse date: {date_str}")
        return datetime.now().strftime("%Y-%m-%d")
    
    @staticmethod
    def normalize_news_article(article: Dict) -> Dict:
        """
        Normalize a news article to common schema.
        
        Args:
            article: Raw article dictionary from news collector.
        
        Returns:
            Normalized article dictionary.
        """
        return {
            "source_type": "news",
            "source_name": article.get("source", "Unknown"),
            "date": DataNormalizer._parse_date(article.get("date", "")),
            "content": article.get("content", ""),
            "title": article.get("title", ""),
            "metadata": {
                "url": article.get("url", ""),
                "author": article.get("author", "Unknown")
            }
        }
    
    @staticmethod
    def normalize_financial_data(data: Dict) -> Dict:
        """
        Normalize financial/economic data to common schema.
        
        Args:
            data: Raw data dictionary from financial collector.
        
        Returns:
            Normalized data dictionary.
        """
        # Check if it's stock price data or economic indicator
        if "ticker" in data:
            # Stock price data
            source = data.get("source", "yfinance")
            return {
                "source_type": "financial",
                "source_name": f"{source} - {data.get('ticker', 'Unknown')}",
                "date": DataNormalizer._parse_date(data.get("date", "")),
                "content": f"Price: ${data.get('price', 0):.2f}",
                "metadata": {
                    "data_type": "stock_price",
                    "ticker": data.get("ticker", ""),
                    "price": data.get("price", 0),
                    "open": data.get("open", 0),
                    "high": data.get("high", 0),
                    "low": data.get("low", 0),
                    "volume": data.get("volume", 0)
                }
            }
        else:
            # Economic indicator
            return {
                "source_type": "financial",
                "source_name": f"FRED - {data.get('indicator', 'Unknown')}",
                "date": DataNormalizer._parse_date(data.get("date", "")),
                "content": f"{data.get('indicator', '')}: {data.get('value', 0):.2f}",
                "metadata": {
                    "data_type": "economic_indicator",
                    "indicator": data.get("indicator", ""),
                    "value": data.get("value", 0)
                }
            }
    
    @staticmethod
    def normalize_job_data(data: Dict) -> Dict:
        """
        Normalize job data to common schema.
        
        Args:
            data: Raw job data dictionary from job collector.
        
        Returns:
            Normalized job data dictionary.
        """
        source_label = data.get("source", "USAJOBS")
        job_title = data.get("job_title", data.get("position_title", "Unknown"))
        company = data.get("company", data.get("organization_name", "Unknown"))
        
        return {
            "source_type": "jobs",
            "source_name": f"{source_label} - {job_title}",
            "date": DataNormalizer._parse_date(data.get("posting_date", data.get("date", ""))),
            "content": data.get("job_description", data.get("position_title", "")),
            "title": job_title,
            "metadata": {
                "company": company,
                "location": data.get("location", data.get("location_name", "")),
                "salary_min": data.get("salary_min"),
                "salary_max": data.get("salary_max"),
                "salary_currency": data.get("salary_currency", "USD"),
                "url": data.get("url", data.get("position_uri", "")),
                "source": source_label
            }
        }
    
    @staticmethod
    def normalize_social_post(data: Dict) -> Dict:
        """
        Normalize social media data from multiple sources to common schema.
        Handles: Reddit (Pushshift), YouTube, Mastodon, HackerNews
        
        Args:
            data: Raw social post dictionary from social collector.
        
        Returns:
            Normalized social data dictionary.
        """
        # Detect source from available fields
        if "subreddit" in data:
            # Reddit/Pushshift
            source_name = f"Reddit - r/{data.get('subreddit', 'unknown')}"
            metadata = {
                "author": data.get("author", "unknown"),
                "subreddit": data.get("subreddit", "unknown"),
                "score": data.get("score", 0),
                "num_comments": data.get("num_comments", 0),
                "url": data.get("url", "")
            }
        elif "video_id" in data:
            # YouTube
            source_name = "YouTube"
            metadata = {
                "author": data.get("author", "unknown"),
                "channel": data.get("channel", "unknown"),
                "video_id": data.get("video_id", ""),
                "video_title": data.get("video_title", ""),
                "likes": data.get("likes", 0),
                "replies": data.get("replies", 0),
                "url": data.get("url", "")
            }
        elif "instance" in data:
            # Mastodon
            source_name = f"Mastodon - {data.get('instance', 'unknown')}"
            metadata = {
                "author": data.get("author", "unknown"),
                "instance": data.get("instance", "unknown"),
                "favorites": data.get("favorites", 0),
                "reblogs": data.get("reblogs", 0),
                "replies": data.get("replies", 0),
                "hashtag": data.get("hashtag", ""),
                "url": data.get("url", "")
            }
        elif data.get("source") == "hackernews_story" or "story_id" in data:
            # HackerNews
            source_name = "HackerNews"
            metadata = {
                "author": data.get("author", "unknown"),
                "story_id": data.get("story_id", ""),
                "points": data.get("points", 0),
                "num_comments": data.get("num_comments", 0),
                "url": data.get("url", ""),
                "source": "hackernews"
            }
        else:
            # Default/unknown social source
            source_name = "Social Media"
            metadata = {
                "author": data.get("author", "unknown"),
                "url": data.get("url", "")
            }
        
        return {
            "source_type": "social",
            "source_name": source_name,
            "date": DataNormalizer._parse_date(data.get("date", "")),
            "content": data.get("content", ""),
            "title": data.get("title", ""),
            "metadata": metadata
        }
    
    @classmethod
    def normalize_all(cls, news: List[Dict], financial: List[Dict], jobs: List[Dict] = None, social: List[Dict] = None) -> List[Dict]:
        """
        Normalize all collected data from all sources.
        
        Args:
            news: List of news articles.
            financial: List of financial data.
            jobs: List of job data (optional).
            social: List of social media records (optional).
        
        Returns:
            Unified list of normalized data records.
        """
        if jobs is None:
            jobs = []
        if social is None:
            social = []
        
        normalized = []
        
        # Normalize news
        for article in news:
            try:
                normalized.append(cls.normalize_news_article(article))
            except Exception as e:
                logger.error(f"Failed to normalize news article: {e}")
                continue
        
        logger.info(f"Normalized {len(normalized)} news articles")
        
        # Normalize financial data
        for record in financial:
            try:
                normalized.append(cls.normalize_financial_data(record))
            except Exception as e:
                logger.error(f"Failed to normalize financial data: {e}")
                continue
        
        logger.info(f"Normalized {len(financial)} financial records")
        
        # Normalize job data
        for record in jobs:
            try:
                normalized.append(cls.normalize_job_data(record))
            except Exception as e:
                logger.error(f"Failed to normalize job data: {e}")
                continue
        
        logger.info(f"Normalized {len(jobs)} job records")
        
        # Normalize social data
        for record in social:
            try:
                normalized.append(cls.normalize_social_post(record))
            except Exception as e:
                logger.error(f"Failed to normalize social data: {e}")
                continue
        
        logger.info(f"Normalized {len(social)} social records")
        logger.info(f"Total normalized records: {len(normalized)}")
        
        return normalized
