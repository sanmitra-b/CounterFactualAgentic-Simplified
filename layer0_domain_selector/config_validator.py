"""
Configuration validator for Layer 1 (moved to Layer 0).
Performs lightweight checks so Layer 0's generated config is consumable by Layer 1.
"""
from typing import Any, Dict, List


class ConfigValidationError(RuntimeError):
    pass


def _ensure_key(d: Dict[str, Any], key: str, parent: str = "config") -> None:
    if key not in d:
        raise ConfigValidationError(f"Missing key '{key}' in {parent}")


def _is_bool_like(v: Any) -> bool:
    return isinstance(v, bool)


def validate_config(config: Dict[str, Any]) -> None:
    """Validate basic shape and required keys of the layer1 config.

    Raises ConfigValidationError on failure.
    """
    if not isinstance(config, dict):
        raise ConfigValidationError("Config must be a JSON object (dict)")

    _ensure_key(config, "sources")
    _ensure_key(config, "output")
    _ensure_key(config, "logging")

    sources = config["sources"]
    if not isinstance(sources, dict):
        raise ConfigValidationError("'sources' must be an object/dictionary")

    # Expected top-level sources and some required method keys
    expected = {
        "news": ["methods"],
        "financial": ["alpha_vantage", "fred"],
        "jobs": ["adzuna", "usajobs"],
        "weather": ["openweather"],
        "social": ["pushshift", "youtube", "mastodon", "hackernews"],
    }

    for src, required_children in expected.items():
        if src not in sources:
            raise ConfigValidationError(f"Missing source section '{src}' in config")
        section = sources[src]
        if not isinstance(section, dict):
            raise ConfigValidationError(f"Source '{src}' must be an object/dictionary")

        # Ensure enabled is present and boolean-like
        if "enabled" not in section:
            raise ConfigValidationError(f"Missing 'enabled' flag for source '{src}'")
        if not _is_bool_like(section["enabled"]):
            raise ConfigValidationError(f"'enabled' for source '{src}' must be boolean")

        # Methods / child keys
        for child in required_children:
            if child not in section:
                raise ConfigValidationError(f"Source '{src}' missing expected child '{child}'")
            # child can be dict or object
            if not isinstance(section[child], (dict, list)):
                raise ConfigValidationError(f"'{child}' under source '{src}' must be an object or list")

    # If weather enabled, ensure cities is a list
    weather = sources.get("weather", {})
    if weather.get("enabled"):
        cities = weather.get("cities")
        if cities is None:
            raise ConfigValidationError("'cities' must be present when weather is enabled")
        if not isinstance(cities, list):
            raise ConfigValidationError("'cities' must be a list of city names")

    # output.domain must be present
    out = config.get("output", {})
    if not out.get("domain"):
        raise ConfigValidationError("'output.domain' must be set in the config")

    # Passed validation
    return None
