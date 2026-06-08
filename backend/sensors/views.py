# backend/sensors/views.py
"""
Dashboard view — the main landing page of Rakshak.

Renders the Operations Control Center dashboard with:
  - 5 KPI summary cards
  - Sensor trend chart data (passed to Chart.js via json_script)
  - Track section health table
  - Recent sensor readings
"""

import json

from django.shortcuts import render

from .mock_data import KPI_SUMMARY, SENSOR_TRENDS, TRACK_SECTIONS, RECENT_READINGS


def dashboard(request):
    """Render the main dashboard page."""
    context = {
        'page_title': 'Dashboard',
        'kpi': KPI_SUMMARY,
        'track_sections': TRACK_SECTIONS,
        'recent_readings': RECENT_READINGS,
        # Serialize trend data for Chart.js (consumed via json_script in template)
        'sensor_trends_json': SENSOR_TRENDS,
    }
    return render(request, 'dashboard.html', context)
