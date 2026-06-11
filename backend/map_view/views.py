# backend/map_view/views.py
"""
Map page view — renders the railway map template.

All map data is now fetched from API endpoints (/api/stations/,
/api/routes/, etc.) by the frontend JavaScript. This view only
renders the HTML template shell.
"""

from django.shortcuts import render


def map_page(request):
    """Render the railway map page."""
    context = {
        "page_title": "Railway Map",
    }
    return render(request, "map.html", context)
