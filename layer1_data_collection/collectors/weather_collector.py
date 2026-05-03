"""
Weather Data Collector Module
Fetches current weather from OpenWeather for configured cities.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List

logger = logging.getLogger(__name__)


DEFAULT_CITIES = [
    "New York",
    "London",
    "Tokyo",
    "Shanghai",
    "Singapore",
    "Hong Kong",
    "Sydney",
    "Dubai",
]


class WeatherCollector:
    """Collects weather signals from OpenWeather."""

    def __init__(self, config: Dict):
        self.config = config
        self.data = []
        self.api_key = os.getenv("OPENWEATHER_API_KEY", "").strip()

    def collect_from_openweather(self) -> List[Dict]:
        """
        Fetch current weather for configured cities.

        Returns:
            List of weather record dictionaries.
        """
        if not self.config.get("openweather", {}).get("enabled"):
            logger.info("OpenWeather disabled in config")
            return []

        if not self.api_key:
            logger.warning("OPENWEATHER_API_KEY not set in .env. Skipping OpenWeather collection.")
            return []

        try:
            import requests
        except ImportError:
            logger.error("requests library not installed")
            return []

        cities = self.config.get("cities") or DEFAULT_CITIES
        units = self.config.get("openweather", {}).get("units", "metric")
        language = self.config.get("openweather", {}).get("language", "en")
        base_url = self.config.get("openweather", {}).get(
            "base_url", "https://api.openweathermap.org/data/2.5/weather"
        )

        weather_data = []

        for city in cities:
            try:
                response = requests.get(
                    base_url,
                    params={
                        "q": city,
                        "appid": self.api_key,
                        "units": units,
                        "lang": language,
                    },
                    timeout=15,
                )
                response.raise_for_status()

                payload = response.json()
                weather = payload.get("weather", [{}])[0]
                main = payload.get("main", {})
                wind = payload.get("wind", {})
                sys_info = payload.get("sys", {})
                coord = payload.get("coord", {})

                weather_data.append(
                    {
                        "city": payload.get("name", city),
                        "country": sys_info.get("country", ""),
                        "date": datetime.utcnow().isoformat(),
                        "temperature": main.get("temp", 0),
                        "wind_speed": wind.get("speed", 0),
                        "weather_main": weather.get("main", ""),
                        "weather_description": weather.get("description", ""),
                        "humidity": main.get("humidity", 0),
                        "pressure": main.get("pressure", 0),
                        "clouds": payload.get("clouds", {}).get("all", 0),
                        "lat": coord.get("lat", 0),
                        "lon": coord.get("lon", 0),
                        "source": "openweather",
                    }
                )
                logger.info(f"OpenWeather: Retrieved weather data for {city}")

            except requests.exceptions.RequestException as error:
                logger.error(f"OpenWeather request failed for {city}: {error}")
                continue
            except Exception as error:
                logger.error(f"Failed to process OpenWeather data for {city}: {error}")
                continue

        return weather_data

    def collect(self) -> List[Dict]:
        logger.info("Starting weather collection...")
        self.data = self.collect_from_openweather()
        logger.info(f"Weather collection complete. Total records: {len(self.data)}")
        return self.data


def collect_weather(config: Dict) -> List[Dict]:
    """
    Convenience function to collect weather.

    Args:
        config: Dictionary with weather configuration.

    Returns:
        List of weather data dictionaries.
    """
    collector = WeatherCollector(config)
    return collector.collect()