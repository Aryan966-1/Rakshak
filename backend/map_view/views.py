from django.shortcuts import render

from .mock_data import STATIONS, RAIL_ROUTES

def map_page(request):
    """Render the railway map page."""
    context = {
        "page_title": "Railway Map",

        # Pass raw Python objects
        # json_script in the template will serialize safely.
        "stations": STATIONS,
        "routes": RAIL_ROUTES,
    }

    return render(
        request,
        "map.html",
        context,
    )
