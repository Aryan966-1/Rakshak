# backend/map_view/services.py
"""
Railway geometry service — loads OpenStreetMap GeoJSON railway data
and converts it into Leaflet-compatible coordinate arrays.

The GeoJSON file uses [longitude, latitude] ordering per the GeoJSON
specification (RFC 7946). Leaflet expects [latitude, longitude].
This module handles the conversion transparently.

Geometry is loaded once at module level and cached for the lifetime
of the process, avoiding repeated disk I/O on every request.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Path to the GeoJSON export (relative to this file)
# ---------------------------------------------------------------------------
_GEOJSON_PATH: Path = Path(__file__).resolve().parent / "route_geometry" / "india_railways.geojson"

# ---------------------------------------------------------------------------
# Module-level cache — populated on first call, reused thereafter
# ---------------------------------------------------------------------------
_cached_geometry: List[List[List[float]]] | None = None


def _flip_coords(coords: list[list[float]]) -> list[list[float]]:
    """
    Convert a list of GeoJSON [lng, lat] pairs to Leaflet [lat, lng] pairs.

    Each coordinate must have at least two elements. The optional third
    element (altitude) is silently discarded since Leaflet doesn't use it.
    """
    return [[lat, lng] for lng, lat, *_ in coords]


def load_railway_geometry() -> List[List[List[float]]]:
    """
    Read the India railways GeoJSON file, extract every LineString
    feature, and return a list of coordinate arrays in Leaflet order.

    Returns:
        A list of routes, where each route is a list of [lat, lng] pairs.
        Example::

            [
                [[28.67, 77.33], [28.67, 77.34], ...],   # route 1
                [[26.94, 76.39], [26.95, 76.40], ...],   # route 2
                ...
            ]

    Raises:
        No exceptions are raised to callers. On any error an empty list
        is returned and a warning is logged so the map gracefully
        degrades to showing only station markers.
    """
    global _cached_geometry

    # Return cached data if available
    if _cached_geometry is not None:
        return _cached_geometry

    try:
        # ------------------------------------------------------------------
        # 1. Read GeoJSON from disk
        # ------------------------------------------------------------------
        if not _GEOJSON_PATH.is_file():
            logger.warning(
                "Railway GeoJSON file not found at %s — "
                "routes will fall back to station-only coordinates.",
                _GEOJSON_PATH,
            )
            _cached_geometry = []
            return _cached_geometry

        with open(_GEOJSON_PATH, "r", encoding="utf-8") as fh:
            data: dict = json.load(fh)

        # ------------------------------------------------------------------
        # 2. Validate top-level structure
        # ------------------------------------------------------------------
        features: list[dict] = data.get("features", [])

        if not features:
            logger.warning(
                "GeoJSON at %s contains an empty FeatureCollection — "
                "no railway geometry will be rendered.",
                _GEOJSON_PATH,
            )
            _cached_geometry = []
            return _cached_geometry

        # ------------------------------------------------------------------
        # 3. Extract & convert LineString geometries
        # ------------------------------------------------------------------
        routes: List[List[List[float]]] = []
        skipped: int = 0

        for feature in features:
            geometry: dict | None = feature.get("geometry")

            # Skip features with missing or null geometry
            if not geometry:
                skipped += 1
                continue

            geom_type: str = geometry.get("type", "")
            raw_coords: list | None = geometry.get("coordinates")

            if geom_type == "LineString" and raw_coords:
                # Each coordinate must be a pair of numbers at minimum
                try:
                    leaflet_coords = _flip_coords(raw_coords)
                    if len(leaflet_coords) >= 2:
                        routes.append(leaflet_coords)
                    else:
                        skipped += 1
                except (TypeError, ValueError):
                    skipped += 1
                    continue

            elif geom_type == "MultiLineString" and raw_coords:
                # Handle MultiLineString by treating each segment separately
                for segment in raw_coords:
                    try:
                        leaflet_coords = _flip_coords(segment)
                        if len(leaflet_coords) >= 2:
                            routes.append(leaflet_coords)
                        else:
                            skipped += 1
                    except (TypeError, ValueError):
                        skipped += 1
                        continue
            else:
                # Point, Polygon, etc. — not relevant for rail routes
                skipped += 1

        if skipped:
            logger.info(
                "Loaded %d railway segments from GeoJSON; skipped %d "
                "features (non-LineString or invalid geometry).",
                len(routes),
                skipped,
            )
        else:
            logger.info(
                "Loaded %d railway segments from GeoJSON.", len(routes)
            )

        _cached_geometry = routes
        return _cached_geometry

    except json.JSONDecodeError as exc:
        logger.error(
            "Failed to parse GeoJSON at %s: %s — "
            "routes will fall back to station-only coordinates.",
            _GEOJSON_PATH,
            exc,
        )
        _cached_geometry = []
        return _cached_geometry

    except OSError as exc:
        logger.error(
            "I/O error reading GeoJSON at %s: %s — "
            "routes will fall back to station-only coordinates.",
            _GEOJSON_PATH,
            exc,
        )
        _cached_geometry = []
        return _cached_geometry
