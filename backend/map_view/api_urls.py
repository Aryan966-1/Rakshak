# map_view/api_urls.py
"""
API URL routing for the map view.

All endpoints return JSON and are consumed by the Leaflet
frontend via fetch() calls in map.js.
"""

from django.urls import path

from . import api_views

app_name = "map_api"

urlpatterns = [
    path("stations/", api_views.api_stations, name="stations"),
    path("routes/",   api_views.api_routes,   name="routes"),
    path("tickets/",  api_views.api_tickets,  name="tickets"),
    path("alerts/",   api_views.api_alerts,   name="alerts"),
    path("summary/",  api_views.api_summary,  name="summary"),
    path("trains/",   api_views.api_trains,   name="trains"),
]
