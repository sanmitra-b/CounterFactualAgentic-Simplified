"""
Job Data Collector Module
Fetches job data from Adzuna and USAJOBS APIs.
"""

import logging
import os
import time
from typing import List, Dict
from pathlib import Path

logger = logging.getLogger(__name__)

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    load_dotenv(env_path)
except ImportError:
    logger.warning("python-dotenv not installed, using environment variables directly")

# Adzuna API Rate Limits (Free Tier)
# Per Minute: 25 hits (1 call/second max)
# Per Day: 250 hits
# Per Week: 1,000 hits
# Per Month: 2,500 hits
ADZUNA_RATE_LIMIT_DELAY = 1.0  # seconds between requests (1 call/second max)


class JobCollector:
    """Collects job data from Adzuna and USAJOBS APIs."""
    
    def __init__(self, config: Dict):
        """
        Initialize the job collector with configuration.
        
        Args:
            config: Dictionary containing configuration from config.json
        """
        self.config = config
        self.data = []
        self.adzuna_id = os.getenv("ADZUNA_ID", "").strip()
        self.adzuna_key = os.getenv("ADZUNA_API_KEY", "").strip()
        self.usajobs_key = os.getenv("USAJOBS_API_KEY", "").strip()
    
    def collect_from_adzuna(self) -> List[Dict]:
        """
        Fetch job data from Adzuna API.
        
        Returns:
            List of job data dictionaries.
        """
        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return []
        
        job_data = []
        
        # Adzuna API endpoint (requires page number in path for this API version)
        base_url = "https://api.adzuna.com/v1/api/jobs/us/search/1"
        
        search_terms = (
            self.config.get("keywords")
            or self.config.get("usajobs", {}).get("keywords")
            or ["financial analyst", "economist", "data scientist", "risk analyst", "actuary"]
        )

        if not self.adzuna_id or not self.adzuna_key:
            logger.warning("ADZUNA_ID or ADZUNA_API_KEY not set in .env. Skipping Adzuna collection.")
            return job_data
        
        for search_term in search_terms:
            try:
                params = {
                    "app_id": self.adzuna_id,
                    "app_key": self.adzuna_key,
                    "what": search_term,
                    "results_per_page": 10
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                
                # Handle rate limiting (429 Too Many Requests)
                if response.status_code == 429:
                    logger.warning(f"Adzuna rate limit exceeded for '{search_term}'. Waiting 60 seconds before retry...")
                    time.sleep(60)
                    response = requests.get(base_url, params=params, timeout=10)
                
                response.raise_for_status()
                
                data = response.json()
                
                if "results" in data:
                    for result in data["results"]:
                        job_data.append({
                            "job_title": result.get("title", ""),
                            "company": result.get("company", {}).get("display_name", ""),
                            "location": result.get("location", {}).get("display_name", ""),
                            "salary_min": result.get("salary_min", None),
                            "salary_max": result.get("salary_max", None),
                            "salary_currency": result.get("salary_currency", ""),
                            "job_description": result.get("description", ""),
                            "posting_date": result.get("created", ""),
                            "url": result.get("redirect_url", ""),
                            "source": "Adzuna",
                            "search_term": search_term
                        })
                    
                    logger.info(f"Adzuna: Retrieved {len(data['results'])} jobs for '{search_term}'")
                else:
                    logger.warning(f"Adzuna: No results for '{search_term}'")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Adzuna request failed for '{search_term}': {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process Adzuna data for '{search_term}': {e}")
                continue
            finally:
                # Respect Adzuna API rate limit (1 call/second max)
                time.sleep(ADZUNA_RATE_LIMIT_DELAY)
        
        return job_data
    
    def collect_from_usajobs(self) -> List[Dict]:
        """
        Fetch job data from USAJOBS API.
        
        Returns:
            List of job data dictionaries.
        """
        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return []
        
        job_data = []
        base_url = "https://data.usajobs.gov/api/search"
        
        # Search for various federal job categories
        configured_keywords = (
            self.config.get("keywords")
            or self.config.get("usajobs", {}).get("keywords")
            or ["economist", "financial analyst", "risk analyst", "data scientist"]
        )

        if not self.usajobs_key:
            logger.warning("USAJOBS_API_KEY not set in .env. Skipping USAJOBS collection.")
            return job_data

        search_params_list = [
            {"keyword": keyword, "agency": "TR" if i % 2 == 0 else "CM"}
            for i, keyword in enumerate(configured_keywords)
        ]
        
        headers = {
            "Authorization-Key": self.usajobs_key,
            "User-Agent": "CounterFactualAgentic-DataCollector/1.0"
        }
        
        for search_param in search_params_list:
            try:
                params = {
                    "Keyword": search_param["keyword"],
                    "AgencyID": search_param["agency"],
                    "ResultsPerPage": "10",
                    "Page": "1"
                }
                
                response = requests.get(base_url, params=params, headers=headers, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                if "SearchResult" in data and "SearchResultItems" in data["SearchResult"]:
                    for item in data["SearchResult"]["SearchResultItems"]:
                        job_info = item.get("MatchedObjectDescriptor", {})
                        job_data.append({
                            "job_title": job_info.get("PositionTitle", ""),
                            "company": job_info.get("DepartmentName", ""),
                            "location": job_info.get("JobLocation", [{}])[0].get("LocationName", "") if job_info.get("JobLocation") else "",
                            "salary_min": job_info.get("PositionOfferingType", [{}])[0].get("RateIntervalCode", None) if job_info.get("PositionOfferingType") else None,
                            "salary_max": None,
                            "job_description": job_info.get("UserArea", {}).get("Details", {}).get("JobSummary", ""),
                            "posting_date": job_info.get("PublicationStartDate", ""),
                            "url": item.get("MatchedObjectDescriptor", {}).get("ApplyURI", [None])[0] if item.get("MatchedObjectDescriptor", {}).get("ApplyURI") else "",
                            "source": "USAJOBS",
                            "search_term": search_param["keyword"]
                        })
                    
                    logger.info(f"USAJOBS: Retrieved {len(data['SearchResult']['SearchResultItems'])} jobs for '{search_param['keyword']}'")
                else:
                    logger.warning(f"USAJOBS: No results for '{search_param['keyword']}'")
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"USAJOBS request failed for '{search_param['keyword']}': {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process USAJOBS data for '{search_param['keyword']}': {e}")
                continue
        
        return job_data
    
    def collect(self) -> List[Dict]:
        """
        Collect job data from all enabled sources (Adzuna and USAJOBS).
        
        Note: Respects Adzuna free tier rate limits:
          - Per Minute: 25 hits (1 call/second max)
          - Per Day: 250 hits
          - Per Week: 1,000 hits
          - Per Month: 2,500 hits
        
        Returns:
            List of all collected job data.
        """
        logger.info("Starting job data collection...")
        logger.info(f"Adzuna rate limit: {ADZUNA_RATE_LIMIT_DELAY}s delay between requests (free tier: 1 call/sec max)")
        
        adzuna_data = self.collect_from_adzuna()
        usajobs_data = self.collect_from_usajobs()
        
        all_data = adzuna_data + usajobs_data
        
        self.data = all_data
        logger.info(f"Job collection complete. Total records: {len(self.data)} (Adzuna: {len(adzuna_data)}, USAJOBS: {len(usajobs_data)})")
        
        return self.data


def collect_jobs(config: Dict) -> List[Dict]:
    """
    Convenience function to collect job data from Adzuna and USAJOBS.
    
    Args:
        config: Dictionary with job configuration.
    
    Returns:
        List of job data dictionaries.
    """
    collector = JobCollector(config)
    return collector.collect()
