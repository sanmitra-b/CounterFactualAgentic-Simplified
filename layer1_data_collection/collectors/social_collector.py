"""
Social Data Collector Module
Fetches social posts from Pushshift Reddit archive.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

logger = logging.getLogger(__name__)


class SocialCollector:
    """Collects Reddit social data via Pushshift API."""

    def __init__(self, config: Dict):
        self.config = config
        self.data = []

    def collect_from_pushshift(self) -> List[Dict]:
        """
        Fetch social posts from Pushshift based on keywords/subreddits.

        Returns:
            List of social post dictionaries.
        """
        pushshift_cfg = self.config.get("pushshift", {})
        if not pushshift_cfg.get("enabled"):
            logger.info("Pushshift disabled in config")
            return []

        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return []

        base_url = pushshift_cfg.get(
            "base_url", "https://api.pushshift.io/reddit/search/submission/"
        )
        fallback_urls = pushshift_cfg.get("fallback_urls", [])
        endpoint_urls = [base_url] + [url for url in fallback_urls if url and url != base_url]
        headers = {"User-Agent": "CFASimplified/1.0 (+social collector)"}
        subreddits = pushshift_cfg.get(
            "subreddits", ["stocks", "investing", "economy", "finance"]
        )
        keywords = pushshift_cfg.get("keywords", ["market", "inflation", "recession"])
        max_per_query = int(pushshift_cfg.get("max_results_per_query", 25))
        days_back = int(pushshift_cfg.get("days_back", 7))

        after_ts = int((datetime.utcnow() - timedelta(days=days_back)).timestamp())
        posts = []

        for subreddit in subreddits:
            for keyword in keywords:
                try:
                    params = {
                        "subreddit": subreddit,
                        "q": keyword,
                        "size": max_per_query,
                        "after": after_ts,
                        "sort": "desc",
                        "sort_type": "created_utc",
                    }

                    def fetch_entries(request_params: Dict):
                        entries_local = []
                        used_url_local = None
                        for endpoint_url in endpoint_urls:
                            try:
                                response = requests.get(
                                    endpoint_url,
                                    params=request_params,
                                    headers=headers,
                                    timeout=15,
                                    allow_redirects=True,
                                )
                                response.raise_for_status()

                                payload = response.json()
                                entries_local = payload.get("data", [])
                                used_url_local = endpoint_url
                                break
                            except requests.exceptions.RequestException:
                                continue
                        return entries_local, used_url_local

                    entries, used_url = fetch_entries(params)

                    # Some mirrors can have sparse recent indexing; retry without the strict 'after' filter.
                    if not entries:
                        relaxed_params = {
                            "subreddit": subreddit,
                            "q": keyword,
                            "size": max_per_query,
                            "sort": "desc",
                            "sort_type": "created_utc",
                        }
                        entries, used_url = fetch_entries(relaxed_params)

                    if used_url is None:
                        raise requests.exceptions.HTTPError("All Pushshift endpoints failed")

                    for entry in entries:
                        created_utc = entry.get("created_utc")
                        if created_utc:
                            date_str = datetime.utcfromtimestamp(created_utc).isoformat()
                        else:
                            date_str = datetime.utcnow().isoformat()

                        posts.append(
                            {
                                "title": entry.get("title", ""),
                                "content": entry.get("selftext", ""),
                                "date": date_str,
                                "author": entry.get("author", "unknown"),
                                "subreddit": entry.get("subreddit", subreddit),
                                "score": entry.get("score", 0),
                                "num_comments": entry.get("num_comments", 0),
                                "url": entry.get("full_link")
                                or entry.get("url")
                                or "",
                            }
                        )

                    logger.info(
                        f"Pushshift: Found {len(entries)} posts in r/{subreddit} for '{keyword}' via {used_url}"
                    )

                except requests.exceptions.RequestException as e:
                    logger.error(
                        f"Pushshift request failed for r/{subreddit}, keyword '{keyword}': {e}"
                    )
                    continue
                except Exception as e:
                    logger.error(
                        f"Failed to process Pushshift data for r/{subreddit}, keyword '{keyword}': {e}"
                    )
                    continue

        return posts

    def collect(self) -> List[Dict]:
        logger.info("Starting social data collection...")
        self.data = self.collect_from_pushshift()
        logger.info(f"Social collection complete. Total records: {len(self.data)}")
        return self.data


def collect_social(config: Dict) -> List[Dict]:
    collector = SocialCollector(config)
    return collector.collect()
