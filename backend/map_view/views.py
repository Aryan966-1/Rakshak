from django.shortcuts import render

from .mock_data import STATIONS, RAIL_ROUTES, MAP_SUMMARY

def map_page(request):
    """Render the railway map page."""
    context = {
        "page_title": "Railway Map",

        # Pass raw Python objects
        # json_script in the template will serialize safely.
        "stations": STATIONS,
        "routes": RAIL_ROUTES,
        "summary": MAP_SUMMARY,
    }

    return render(
        request,
        "map.html",
        context,
    )
