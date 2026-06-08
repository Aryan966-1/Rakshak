# backend/map_view/views.py
"""
Map view — displays the railway network on a Leaflet map.

Renders an interactive map centered on India with station markers,
alert indicators, and route polylines.
"""

import json

from django.shortcuts import render

from .mock_data import STATIONS, RAIL_ROUTES


def map_page(request):
    """Render the railway map page."""
    context = {
        'page_title': 'Railway Map',
        # Serialize map data for Leaflet (consumed via json_script in template)
        'stations_json': json.dumps(STATIONS),
        'routes_json': json.dumps(RAIL_ROUTES),
    }
    return render(request, 'map.html', context)
