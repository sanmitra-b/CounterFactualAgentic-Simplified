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
                            try:
                                # Handle both int and string timestamps
                                ts = int(created_utc) if isinstance(created_utc, str) else created_utc
                                date_str = datetime.utcfromtimestamp(ts).isoformat()
                            except (ValueError, OSError, TypeError):
                                date_str = datetime.utcnow().isoformat()
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

    def collect_from_youtube(self) -> List[Dict]:
        """
        Fetch comments from YouTube videos about AI/jobs using YouTube Data API v3.

        Returns:
            List of YouTube comment dictionaries.
        """
        youtube_cfg = self.config.get("youtube", {})
        if not youtube_cfg.get("enabled"):
            logger.info("YouTube disabled in config")
            return []

        try:
            from googleapiclient.discovery import build
        except ImportError:
            logger.warning("google-api-python-client not installed. Install with: pip install google-api-python-client")
            return []

        api_key = youtube_cfg.get("api_key", "").strip()
        if api_key == "YOUR_YOUTUBE_API_KEY" or not api_key:
            logger.warning("YouTube API key not configured or set to placeholder")
            return []

        try:
            youtube = build("youtube", "v3", developerKey=api_key)
            keywords = youtube_cfg.get("keywords", ["AI jobs"])
            max_results = int(youtube_cfg.get("max_results", 50))
            order = youtube_cfg.get("order", "relevance")
            
            comments = []
            
            for keyword in keywords:
                try:
                    # Search for videos
                    search_response = youtube.search().list(
                        q=keyword,
                        part="snippet",
                        type="video",
                        maxResults=5,
                        order=order,
                        textFormat="plainText"
                    ).execute()

                    video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
                    
                    for video_id in video_ids:
                        try:
                            # Get video details
                            video_response = youtube.videos().list(
                                part="snippet,statistics",
                                id=video_id
                            ).execute()
                            
                            video_item = video_response["items"][0] if video_response["items"] else None
                            if not video_item:
                                continue
                                
                            video_title = video_item["snippet"]["title"]
                            channel_title = video_item["snippet"]["channelTitle"]
                            published_at = video_item["snippet"]["publishedAt"]
                            view_count = video_item["statistics"].get("viewCount", 0)
                            
                            # Get top comments
                            comments_response = youtube.commentThreads().list(
                                videoId=video_id,
                                part="snippet",
                                textFormat="plainText",
                                maxResults=20,
                                order="relevance"
                            ).execute()

                            for comment_thread in comments_response.get("items", []):
                                comment = comment_thread["snippet"]["topLevelComment"]["snippet"]
                                comments.append({
                                    "title": f"YouTube: {video_title}",
                                    "content": comment["textDisplay"],
                                    "date": comment["publishedAt"],
                                    "author": comment["authorDisplayName"],
                                    "video_id": video_id,
                                    "channel": channel_title,
                                    "video_title": video_title,
                                    "likes": comment["likeCount"],
                                    "replies": comment.get("replyCount", 0),
                                    "url": f"https://www.youtube.com/watch?v={video_id}"
                                })

                            logger.info(f"YouTube: Collected {len(comments_response.get('items', []))} comments from video: {video_title[:50]}")
                            
                            if len(comments) >= max_results:
                                break
                                
                        except Exception as e:
                            logger.debug(f"Failed to get comments for YouTube video {video_id}: {e}")
                            continue
                    
                    if len(comments) >= max_results:
                        break
                        
                except Exception as e:
                    logger.error(f"YouTube search failed for keyword '{keyword}': {e}")
                    continue
                    
            logger.info(f"YouTube: Collection complete. Total comments: {len(comments)}")
            return comments[:max_results]
            
        except Exception as e:
            logger.error(f"YouTube collection error: {e}")
            return []

    def collect_from_mastodon(self) -> List[Dict]:
        """
        Fetch posts from Mastodon instances using public timeline and hashtag search.

        Returns:
            List of Mastodon post dictionaries.
        """
        mastodon_cfg = self.config.get("mastodon", {})
        if not mastodon_cfg.get("enabled"):
            logger.info("Mastodon disabled in config")
            return []

        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return []

        instances = mastodon_cfg.get("instances", ["fosstodon.org", "techhub.social"])
        keywords = mastodon_cfg.get("keywords", ["AI jobs"])
        hashtags = mastodon_cfg.get("hashtags", ["ai", "jobs"])
        max_per_instance = int(mastodon_cfg.get("max_results_per_instance", 20))
        
        posts = []
        headers = {"User-Agent": "CFASimplified/1.0 (+mastodon collector)"}

        for instance in instances:
            try:
                base_url = f"https://{instance}/api/v1"
                
                # Search for statuses using keywords
                for keyword in keywords:
                    try:
                        search_response = requests.get(
                            f"{base_url}/search",
                            params={"q": keyword, "type": "statuses", "limit": max_per_instance},
                            headers=headers,
                            timeout=10
                        )
                        search_response.raise_for_status()
                        search_data = search_response.json()
                        
                        for status in search_data.get("statuses", []):
                            posts.append({
                                "title": f"Mastodon ({instance}): {keyword}",
                                "content": status.get("content", "").replace("<p>", "").replace("</p>", ""),
                                "date": status.get("created_at", ""),
                                "author": status.get("account", {}).get("acct", "unknown"),
                                "instance": instance,
                                "favorites": status.get("favourites_count", 0),
                                "reblogs": status.get("reblogs_count", 0),
                                "replies": status.get("replies_count", 0),
                                "url": status.get("url", "")
                            })
                        
                        logger.info(f"Mastodon ({instance}): Found {len(search_data.get('statuses', []))} posts for '{keyword}'")
                        
                    except Exception as e:
                        logger.debug(f"Mastodon search failed for {instance}, keyword '{keyword}': {e}")
                        continue
                
                # Also search by hashtags
                for hashtag in hashtags:
                    try:
                        hashtag_response = requests.get(
                            f"{base_url}/timelines/tag/{hashtag}",
                            params={"limit": max_per_instance // 2},
                            headers=headers,
                            timeout=10
                        )
                        hashtag_response.raise_for_status()
                        hashtag_data = hashtag_response.json()
                        
                        for status in hashtag_data:
                            posts.append({
                                "title": f"Mastodon ({instance}): #{hashtag}",
                                "content": status.get("content", "").replace("<p>", "").replace("</p>", ""),
                                "date": status.get("created_at", ""),
                                "author": status.get("account", {}).get("acct", "unknown"),
                                "instance": instance,
                                "hashtag": hashtag,
                                "favorites": status.get("favourites_count", 0),
                                "reblogs": status.get("reblogs_count", 0),
                                "replies": status.get("replies_count", 0),
                                "url": status.get("url", "")
                            })
                        
                        logger.info(f"Mastodon ({instance}): Found {len(hashtag_data)} posts for #{hashtag}")
                        
                    except Exception as e:
                        logger.debug(f"Mastodon hashtag search failed for {instance}, hashtag '{hashtag}': {e}")
                        continue
                        
            except Exception as e:
                logger.error(f"Mastodon collection error for instance {instance}: {e}")
                continue

        logger.info(f"Mastodon: Collection complete. Total posts: {len(posts)}")
        return posts

    def collect_from_hackernews(self) -> List[Dict]:
        """
        Fetch posts and comments from HackerNews about tech/jobs/AI using HackerNews API.

        Returns:
            List of HackerNews item dictionaries.
        """
        hn_cfg = self.config.get("hackernews", {})
        if not hn_cfg.get("enabled"):
            logger.info("HackerNews disabled in config")
            return []

        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return []

        keywords = hn_cfg.get("keywords", ["AI jobs"])
        max_results = int(hn_cfg.get("max_results", 50))
        min_comments = int(hn_cfg.get("min_comments", 5))
        
        items = []
        headers = {"User-Agent": "CFASimplified/1.0 (+hackernews collector)"}
        
        try:
            # HackerNews Algolia API for searching
            algolia_api = "https://hn.algolia.com/api/v1/search"
            
            for keyword in keywords:
                try:
                    # Search for stories
                    response = requests.get(
                        algolia_api,
                        params={
                            "query": keyword,
                            "tags": "story",
                            "numericFilters": f"comments>={min_comments}",
                            "hitsPerPage": 20
                        },
                        headers=headers,
                        timeout=10
                    )
                    response.raise_for_status()
                    results = response.json()
                    
                    for hit in results.get("hits", []):
                        items.append({
                            "title": hit.get("title", ""),
                            "content": hit.get("story_text", "") or f"Link: {hit.get('url', '')}",
                            "date": hit.get("created_at", ""),
                            "author": hit.get("author", "unknown"),
                            "story_id": hit.get("objectID", ""),
                            "url": hit.get("url") or f"https://news.ycombinator.com/item?id={hit.get('objectID', '')}",
                            "points": hit.get("points", 0),
                            "num_comments": hit.get("num_comments", 0),
                            "source": "hackernews_story"
                        })
                    
                    logger.info(f"HackerNews: Found {len(results.get('hits', []))} stories for '{keyword}'")
                    
                    if len(items) >= max_results:
                        break
                        
                except Exception as e:
                    logger.error(f"HackerNews search failed for keyword '{keyword}': {e}")
                    continue
            
            logger.info(f"HackerNews: Collection complete. Total items: {len(items)}")
            return items[:max_results]
            
        except Exception as e:
            logger.error(f"HackerNews collection error: {e}")
            return []

    def collect(self) -> List[Dict]:
        logger.info("Starting social data collection...")
        
        self.data = self.collect_from_pushshift()
        pushshift_count = len(self.data)
        
        youtube_data = self.collect_from_youtube()
        self.data.extend(youtube_data)
        
        mastodon_data = self.collect_from_mastodon()
        self.data.extend(mastodon_data)
        
        hackernews_data = self.collect_from_hackernews()
        self.data.extend(hackernews_data)
        
        logger.info(f"Social collection complete. Total records: {len(self.data)} "
                   f"(Pushshift: {pushshift_count}, YouTube: {len(youtube_data)}, "
                   f"Mastodon: {len(mastodon_data)}, HackerNews: {len(hackernews_data)})")
        
        return self.data


def collect_social(config: Dict) -> List[Dict]:
    collector = SocialCollector(config)
    return collector.collect()
