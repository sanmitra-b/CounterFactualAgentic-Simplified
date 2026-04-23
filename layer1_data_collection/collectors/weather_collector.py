"""
Weather Data Collector Module
Fetches exogenous risk signals from OpenWeather API or USAJOBS API.
"""

import logging
import os
from typing import List, Dict
from datetime import datetime
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except Exception:
    pass

logger = logging.getLogger(__name__)


class WeatherCollector:
    """Collects weather-like risk signals from configured external source APIs."""
    
    def __init__(self, config: Dict):
        """
        Initialize the weather collector with configuration.
        
        Args:
            config: Dictionary containing 'weather' settings from config.json
        """
        self.config = config
        self.data = []
    
    def collect_from_openweather(self) -> List[Dict]:
        """
        Fetch current weather data from OpenWeather API.
        
        Returns:
            List of weather data dictionaries.
        """
        if not self.config.get("openweather", {}).get("enabled"):
            logger.info("OpenWeather disabled in config")
            return []
        
        api_key = self.config["openweather"].get("api_key")
        if not api_key or api_key == "YOUR_KEY_HERE":
            logger.warning("OpenWeather API key not configured. Skipping OpenWeather.")
            return []
        
        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return []
        
        # List of major economic cities to track weather
        cities = self.config["openweather"].get("cities", [
            "New York", "London", "Tokyo", "Shanghai", "Singapore",
            "Hong Kong", "Sydney", "Dubai", "Mexico City", "Toronto"
        ])
        
        weather_data = []
        base_url = "https://api.openweathermap.org/data/2.5/weather"
        
        for city in cities:
            try:
                params = {
                    "q": city,
                    "appid": api_key,
                    "units": "metric"  # Celsius
                }
                
                response = requests.get(base_url, params=params, timeout=10)
                response.raise_for_status()
                
                data = response.json()
                
                # Check for errors
                if "cod" in data and data["cod"] != 200:
                    logger.warning(f"OpenWeather error for {city}: {data.get('message')}")
                    continue
                
                weather_data.append({
                    "city": city,
                    "country": data.get("sys", {}).get("country", ""),
                    "date": datetime.fromtimestamp(data.get("dt", 0)).isoformat(),
                    "temperature": data.get("main", {}).get("temp", 0),
                    "feels_like": data.get("main", {}).get("feels_like", 0),
                    "humidity": data.get("main", {}).get("humidity", 0),
                    "pressure": data.get("main", {}).get("pressure", 0),
                    "weather_main": data.get("weather", [{}])[0].get("main", ""),
                    "weather_description": data.get("weather", [{}])[0].get("description", ""),
                    "wind_speed": data.get("wind", {}).get("speed", 0),
                    "cloudiness": data.get("clouds", {}).get("all", 0),
                    "source": "OpenWeather"
                })
                
                logger.info(f"OpenWeather: Retrieved weather data for {city}")
                
            except requests.exceptions.RequestException as e:
                logger.error(f"OpenWeather request failed for {city}: {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process OpenWeather data for {city}: {e}")
                continue
        
        return weather_data

    def collect_from_usajobs(self) -> List[Dict]:
        """
        Fetch AI-related job market postings from USAJOBS API.

        Returns:
            List of records mapped into the weather-like schema used downstream.
        """
        if not self.config.get("usajobs", {}).get("enabled"):
            logger.info("USAJOBS disabled in config")
            return []

        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return []

        api_key = (
            os.getenv("USAJOBS_API_KEY", "").strip()
            or self.config["usajobs"].get("api_key", "").strip()
        )
        if not api_key:
            logger.warning("USAJOBS API key not configured. Skipping USAJOBS.")
            return []

        base_url = self.config["usajobs"].get("base_url", "https://data.usajobs.gov/api/search")
        user_agent = (
            os.getenv("USAJOBS_USER_AGENT", "").strip()
            or self.config["usajobs"].get("user_agent", "").strip()
            or "ai-job-risk-monitor@example.com"
        )
        keywords = self.config["usajobs"].get("keywords", ["artificial intelligence", "machine learning"])
        results_per_query = int(self.config["usajobs"].get("results_per_query", 25))

        headers = {
            "Host": "data.usajobs.gov",
            "User-Agent": user_agent,
            "Authorization-Key": api_key,
            "Accept": "application/json",
        }

        usajobs_data = []
        for keyword in keywords:
            try:
                params = {
                    "Keyword": keyword,
                    "ResultsPerPage": results_per_query,
                }
                response = requests.get(base_url, headers=headers, params=params, timeout=20)
                response.raise_for_status()
                payload = response.json()

                items = (
                    payload.get("SearchResult", {})
                    .get("SearchResultItems", [])
                )

                for item in items:
                    descriptor = item.get("MatchedObjectDescriptor", {})
                    position_title = descriptor.get("PositionTitle", "")
                    organization = descriptor.get("OrganizationName", "")
                    location = descriptor.get("PositionLocationDisplay", "US")
                    publication_start = descriptor.get("PublicationStartDate", datetime.utcnow().isoformat())
                    publication_end = descriptor.get("ApplicationCloseDate", "")
                    details_url = ""
                    if descriptor.get("PositionURI"):
                        details_url = descriptor.get("PositionURI")

                    remuneration = descriptor.get("PositionRemuneration", [])
                    min_salary = 0.0
                    max_salary = 0.0
                    if remuneration:
                        min_salary = float(remuneration[0].get("MinimumRange", 0) or 0)
                        max_salary = float(remuneration[0].get("MaximumRange", 0) or 0)

                    summary_parts = [
                        f"{position_title}",
                        f"Org: {organization}",
                        f"Keyword: {keyword}",
                    ]
                    if min_salary or max_salary:
                        summary_parts.append(f"Salary: ${min_salary:,.0f}-${max_salary:,.0f}")
                    if publication_end:
                        summary_parts.append(f"Close: {publication_end}")
                    description = " | ".join(part for part in summary_parts if part)

                    usajobs_data.append(
                        {
                            "city": location,
                            "country": "US",
                            "date": publication_start,
                            "temperature": 0.0,
                            "feels_like": 0.0,
                            "humidity": 0.0,
                            "pressure": 0.0,
                            "weather_main": position_title,
                            "weather_description": description,
                            "wind_speed": 0.0,
                            "cloudiness": 0,
                            "source": "USAJOBS",
                            "url": details_url,
                            "keyword": keyword,
                        }
                    )

                logger.info(f"USAJOBS: Retrieved {len(items)} records for keyword '{keyword}'")

            except requests.exceptions.RequestException as e:
                logger.error(f"USAJOBS request failed for keyword '{keyword}': {e}")
                continue
            except Exception as e:
                logger.error(f"Failed to process USAJOBS data for keyword '{keyword}': {e}")
                continue

        return usajobs_data
    
    def collect(self) -> List[Dict]:
        """
        Collect weather data from all enabled sources.
        
        Returns:
            List of all collected weather data.
        """
        logger.info("Starting weather data collection...")

        all_data = []
        all_data.extend(self.collect_from_usajobs())
        all_data.extend(self.collect_from_openweather())
        
        self.data = all_data
        logger.info(f"Weather collection complete. Total records: {len(self.data)}")
        
        return self.data


def collect_weather(config: Dict) -> List[Dict]:
    """
    Convenience function to collect weather data.
    
    Args:
        config: Dictionary with weather configuration.
    
    Returns:
        List of weather data dictionaries.
    """
    collector = WeatherCollector(config)
    return collector.collect()
