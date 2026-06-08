# backend/core/context_processors.py
"""
Shared context processors injected into every template.

These provide navigation items and project metadata so that
base.html can render the nav bar and footer without each view
having to repeat the same context.
"""

from django.utils import timezone


def navigation(request):
    """Inject navigation items with active-page detection."""
    nav_items = [
        {
            'name': 'Dashboard',
            'url': '/',
            'icon': 'dashboard',
            'description': 'System Overview',
        },
        {
            'name': 'Alerts',
            'url': '/alerts/',
            'icon': 'alerts',
            'description': 'Active Alerts',
        },
        {
            'name': 'Tickets',
            'url': '/tickets/',
            'icon': 'tickets',
            'description': 'Maintenance Tickets',
        },
        {
            'name': 'Map',
            'url': '/map/',
            'icon': 'map',
            'description': 'Railway Network',
        },
    ]
    current_path = request.path
    for item in nav_items:
        item['active'] = current_path == item['url']
    return {'nav_items': nav_items}


def project_meta(request):
    """Inject project-wide metadata."""
    return {
        'project_name': 'RAKSHAK',
        'project_subtitle': 'Railway Operations Control Center',
        'project_version': 'Phase 1 — Prototype',
        'server_time': timezone.now(),
    }
