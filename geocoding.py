"""Search for places with the free Open-Meteo geocoding API.

No API key is needed. This module only handles place search; it does not touch
the map or the weather pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

import config


class GeocodingError(Exception):
    """Raised when place search fails."""


@dataclass(frozen=True)
class Place:
    """A place returned by Open-Meteo."""

    name: str
    latitude: float
    longitude: float
    country: str = ""
    admin1: str = ""

    @property
    def label(self) -> str:
        """A readable label such as 'Algiers, Algeria'."""
        parts = [self.name]

        if self.admin1 and self.admin1 != self.name:
            parts.append(self.admin1)

        if self.country:
            parts.append(self.country)

        return ", ".join(parts)


def search_places(
    query: str,
    count: int = config.GEOCODING_RESULT_COUNT,
) -> list[Place]:
    """Search Open-Meteo for up to ``count`` matching places."""
    text = query.strip()
    if not text:
        return []

    params = {"name": text, "count": count}

    try:
        response = requests.get(
            config.GEOCODING_URL,
            params=params,
            timeout=config.GEOCODING_TIMEOUT_SECONDS,
        )
    except requests.Timeout as exc:
        raise GeocodingError(
            "The place search timed out. Please try again."
        ) from exc
    except requests.RequestException as exc:
        raise GeocodingError(
            "Could not reach the place search service. Please try again."
        ) from exc

    if response.status_code != 200:
        raise GeocodingError(
            f"The place search service returned an error "
            f"(HTTP {response.status_code})."
        )

    try:
        payload = response.json()
    except ValueError as exc:
        raise GeocodingError(
            "The place search service returned an unreadable response."
        ) from exc

    results = payload.get("results") or []
    places: list[Place] = []

    for item in results:
        if not isinstance(item, dict):
            continue

        name = str(item.get("name", "")).strip()
        if not name:
            continue

        try:
            latitude = float(item["latitude"])
            longitude = float(item["longitude"])
        except (KeyError, TypeError, ValueError):
            continue

        places.append(
            Place(
                name=name,
                latitude=latitude,
                longitude=longitude,
                country=str(item.get("country", "") or ""),
                admin1=str(item.get("admin1", "") or ""),
            )
        )

    return places
