# backend/rakshak_project/urls.py
"""
Root URL configuration for the Rakshak project.

Routes:
  /            → Dashboard (sensors app)
  /alerts/     → Alerts page
  /tickets/    → Maintenance Tickets page
  /map/        → Railway Map page
  /api/        → JSON API endpoints (map data)
"""

from django.urls import path, include

urlpatterns = [
    path('', include('sensors.urls')),
    path('alerts/', include('alerts.urls')),
    path('tickets/', include('tickets.urls')),
    path('map/', include('map_view.urls')),
    path('api/', include('map_view.api_urls')),
]

