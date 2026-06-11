# backend/alerts/views.py
from django.shortcuts import render
from railway.models import Alert


def alerts_page(request):
    """Display alerts with optional severity filtering and summary statistics."""
    severity_filter = request.GET.get('severity', 'all')

    all_alerts = Alert.objects.select_related(
        'track_section__start_station__division__zone',
        'track_section__end_station',
    )

    # Build summary from the full (unfiltered) queryset
    summary = {
        'total': all_alerts.count(),
        'critical': all_alerts.filter(severity='critical').count(),
        'warning': all_alerts.filter(severity='warning').count(),
        'info': all_alerts.filter(severity='info').count(),
        'active': all_alerts.filter(status='active').count(),
        'resolved': all_alerts.filter(status='resolved').count(),
    }

    # Apply severity filter
    if severity_filter != 'all':
        filtered_qs = all_alerts.filter(severity=severity_filter)
    else:
        filtered_qs = all_alerts

    # Serialize each alert to the dict shape expected by the template
    alerts = []
    for alert in filtered_qs:
        alerts.append({
            'id': alert.alert_code,
            'severity': alert.severity,
            'title': alert.title,
            'description': alert.description,
            'track_id': alert.track_section.section_code,
            'section': f"{alert.track_section.start_station.station_name} — {alert.track_section.end_station.station_name}",
            'station': alert.track_section.start_station.station_name,
            'zone': alert.track_section.start_station.division.zone.name,
            'timestamp': alert.generated_at.strftime('%Y-%m-%d %H:%M:%S'),
            'status': alert.status,
        })

    context = {
        'page_title': 'Alerts',
        'alerts': alerts,
        'summary': summary,
        'current_filter': severity_filter,
    }
    return render(request, 'alerts.html', context)
