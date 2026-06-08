# backend/alerts/views.py
"""
Alerts view — displays all infrastructure alerts.

Renders the alerts page with severity-coded alert cards,
filtering controls, and alert history.
"""

from django.shortcuts import render

from .mock_data import ALERTS, ALERT_SUMMARY


def alerts_page(request):
    """Render the alerts listing page."""
    # Optional severity filter from query string (e.g., ?severity=critical)
    severity_filter = request.GET.get('severity', 'all')

    if severity_filter != 'all':
        filtered_alerts = [a for a in ALERTS if a['severity'] == severity_filter]
    else:
        filtered_alerts = ALERTS

    context = {
        'page_title': 'Alerts',
        'alerts': filtered_alerts,
        'summary': ALERT_SUMMARY,
        'current_filter': severity_filter,
    }
    return render(request, 'alerts.html', context)
