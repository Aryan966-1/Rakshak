"""
map_view/api_views.py
JSON API endpoints for the Leaflet map frontend.

All endpoints return JsonResponse — no template rendering.
The map.js frontend fetches these via fetch() to populate
stations, routes, tickets, alerts, and train simulation data.
"""

import random
import time
from decimal import Decimal

from django.db.models import Q
from django.http import JsonResponse

from railway.models import (
    Alert,
    Division,
    Station,
    Ticket,
    TrackSection,
    Zone,
)


def _decimal_to_float(val):
    """Convert Decimal to float for JSON serialization."""
    return float(val) if isinstance(val, Decimal) else val


def api_stations(request):
    """
    GET /api/stations/

    Returns all active stations with coordinates, zone, division,
    and operational metadata for map marker rendering.
    """
    stations = (
        Station.objects
        .filter(is_active=True)
        .select_related("division__zone")
        .order_by("station_name")
    )

    data = []
    for s in stations:
        # Count active alerts on track sections connected to this station
        active_alerts = Alert.objects.filter(
            track_section__start_station=s,
            status="active",
        ).count() + Alert.objects.filter(
            track_section__end_station=s,
            status="active",
        ).count()

        # Determine station health status from alerts
        if active_alerts >= 3:
            status = "critical"
        elif active_alerts >= 1:
            status = "warning"
        else:
            status = "healthy"

        # Deterministic operational data derived from DB
        tracks_monitored = TrackSection.objects.filter(
            Q(start_station=s) | Q(end_station=s)
        ).count()
        # Stable pseudo-random daily_trains derived from station PK
        daily_trains = (s.pk * 37 % 400) + 50

        data.append({
            "code": s.station_code,
            "name": s.station_name,
            "lat": _decimal_to_float(s.latitude),
            "lng": _decimal_to_float(s.longitude),
            "zone": s.division.zone.name,
            "division": s.division.name,
            "is_junction": s.is_junction,
            "is_terminal": s.is_terminal,
            "status": status,
            "active_alerts": active_alerts,
            "tracks_monitored": tracks_monitored,
            "daily_trains": daily_trains,
        })

    return JsonResponse(data, safe=False)


def api_routes(request):
    """
    GET /api/routes/

    Returns all track sections with their polyline geometry
    for rendering as Leaflet polylines on the map.
    """
    sections = (
        TrackSection.objects
        .select_related("start_station", "end_station")
        .order_by("section_code")
    )

    data = []
    status_map = {
        "active": "healthy",
        "under_maintenance": "warning",
        "closed": "critical",
        "decommissioned": "critical",
    }

    for ts in sections:
        # Use stored geometry if available, otherwise fall back to
        # a straight line between the two station endpoints.
        coords = ts.geometry
        if not coords or len(coords) < 2:
            coords = [
                [
                    _decimal_to_float(ts.start_station.latitude),
                    _decimal_to_float(ts.start_station.longitude),
                ],
                [
                    _decimal_to_float(ts.end_station.latitude),
                    _decimal_to_float(ts.end_station.longitude),
                ],
            ]

        data.append({
            "id": ts.section_code,
            "name": (
                f"{ts.start_station.station_name} — "
                f"{ts.end_station.station_name}"
            ),
            "train": "Indian Railways",
            "source": ts.start_station.station_code,
            "destination": ts.end_station.station_code,
            "distance_km": _decimal_to_float(ts.length_km) if ts.length_km else None,
            "coordinates": coords,
            "status": status_map.get(ts.status, "healthy"),
        })

    return JsonResponse(data, safe=False)


def api_tickets(request):
    """
    GET /api/tickets/

    Returns active tickets with location data for map markers.
    Ticket position is derived from the start station of its
    track section (with slight random offset for visual spread).
    """
    tickets = (
        Ticket.objects
        .exclude(status="closed")
        .select_related(
            "track_section__start_station__division__zone",
            "assigned_team",
        )
        .order_by("-created_at")[:200]  # Cap at 200 for map performance
    )

    random.seed(99)  # Deterministic offsets for consistent rendering

    data = []
    for t in tickets:
        sta = t.track_section.start_station
        div = sta.division
        zone = div.zone

        # Slight random offset so overlapping tickets don't stack
        lat = _decimal_to_float(sta.latitude) + random.uniform(-0.05, 0.05)
        lng = _decimal_to_float(sta.longitude) + random.uniform(-0.05, 0.05)

        data.append({
            "id": t.ticket_code,
            "title": t.title,
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "status": t.status,
            "priority": t.priority,
            "zone": zone.name,
            "division": div.name,
            "station": sta.station_name,
            "team": t.assigned_team.team_name if t.assigned_team else "Unassigned",
            "section": (
                f"{t.track_section.start_station.station_name} — "
                f"{t.track_section.end_station.station_name}"
            ),
        })

    return JsonResponse(data, safe=False)


def api_alerts(request):
    """
    GET /api/alerts/

    Returns active alerts with location data for map markers.
    """
    alerts = (
        Alert.objects
        .filter(status__in=["active", "acknowledged"])
        .select_related(
            "track_section__start_station__division__zone",
        )
        .order_by("-generated_at")[:100]
    )

    data = []
    for a in alerts:
        sta = a.track_section.start_station
        data.append({
            "id": a.alert_code,
            "title": a.title,
            "description": a.description,
            "severity": a.severity,
            "status": a.status,
            "lat": _decimal_to_float(sta.latitude),
            "lng": _decimal_to_float(sta.longitude),
            "zone": sta.division.zone.name,
            "station": sta.station_name,
            "generated_at": a.generated_at.isoformat(),
        })

    return JsonResponse(data, safe=False)


def api_summary(request):
    """
    GET /api/summary/

    Returns aggregate statistics for the map stats bar.
    """
    data = {
        "stations": Station.objects.filter(is_active=True).count(),
        "track_sections": TrackSection.objects.count(),
        "active_routes": TrackSection.objects.filter(status="active").count(),
        "railway_zones": Zone.objects.filter(is_active=True).count(),
        "active_tickets": Ticket.objects.exclude(
            status__in=["closed", "resolved"]
        ).count(),
        "active_alerts": Alert.objects.filter(status="active").count(),
    }
    return JsonResponse(data)


def api_trains(request):
    """
    GET /api/trains/

    Returns simulated train positions along routes.
    Each "train" is assigned to a route and its position is
    calculated from the route geometry + current time offset.
    No real GPS — purely for hackathon demo realism.
    """
    sections = list(
        TrackSection.objects
        .filter(status="active")
        .exclude(geometry=[])
        .values("section_code", "geometry", "length_km")
        .order_by("section_code")
    )

    if not sections:
        return JsonResponse([], safe=False)

    # Pick ~20 routes to have "trains" on them
    random.seed(int(time.time()) // 10)  # Changes every 10 seconds
    train_routes = random.sample(sections, k=min(20, len(sections)))

    trains = []
    for i, route in enumerate(train_routes):
        coords = route["geometry"]
        if not coords or len(coords) < 2:
            continue

        # Calculate position along route based on time
        # Each train moves through its route at a different speed
        t = time.time()
        speed_factor = 0.0001 * (i + 1)
        progress = (t * speed_factor) % 1.0  # 0.0 to 1.0

        # Interpolate position along the polyline
        total_segments = len(coords) - 1
        segment_idx = int(progress * total_segments)
        segment_idx = min(segment_idx, total_segments - 1)
        local_t = (progress * total_segments) - segment_idx

        lat = coords[segment_idx][0] + local_t * (
            coords[segment_idx + 1][0] - coords[segment_idx][0]
        )
        lng = coords[segment_idx][1] + local_t * (
            coords[segment_idx + 1][1] - coords[segment_idx][1]
        )

        trains.append({
            "id": f"TRN-{i+1:03d}",
            "route_id": route["section_code"],
            "lat": round(lat, 6),
            "lng": round(lng, 6),
            "progress": round(progress, 3),
            "speed_kmph": random.randint(60, 160),
        })

    return JsonResponse(trains, safe=False)
