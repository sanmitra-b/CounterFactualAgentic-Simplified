"""
Weather Data Collector Module
Fetches weather data from OpenWeather API.
"""

import logging
from typing import List, Dict
from datetime import datetime

logger = logging.getLogger(__name__)


class WeatherCollector:
    """Collects weather data from OpenWeather API."""
    
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
    
    def collect(self) -> List[Dict]:
        """
        Collect weather data from all enabled sources.
        
        Returns:
            List of all collected weather data.
        """
        logger.info("Starting weather data collection...")
        
        all_data = self.collect_from_openweather()
        
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
