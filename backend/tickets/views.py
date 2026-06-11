# backend/tickets/views.py
from django.shortcuts import render
from railway.models import Ticket


def tickets_page(request):
    status_filter = request.GET.get('status', 'all')

    all_tickets = Ticket.objects.select_related(
        'alert',
        'track_section__start_station__division__zone',
        'track_section__end_station',
        'assigned_team',
    )

    # Build summary from ALL tickets (unfiltered)
    summary = {
        'total': all_tickets.count(),
        'assigned': all_tickets.filter(status='assigned').count(),
        'scheduled': all_tickets.filter(status='scheduled').count(),
        'resolved': all_tickets.filter(status='resolved').count(),
        'critical_priority': all_tickets.filter(priority='critical').count(),
    }

    # Apply status filter for the displayed list
    if status_filter != 'all':
        filtered_qs = all_tickets.filter(status=status_filter)
    else:
        filtered_qs = all_tickets

    # Serialize each ticket to match the template's expected dict shape
    filtered_tickets = []
    for ticket in filtered_qs:
        filtered_tickets.append({
            'id': ticket.ticket_code,
            'linked_alert': ticket.alert.alert_code if ticket.alert else '—',
            'track_id': ticket.track_section.section_code,
            'section': (
                f"{ticket.track_section.start_station.station_name} — "
                f"{ticket.track_section.end_station.station_name}"
            ),
            'station': ticket.track_section.start_station.station_name,
            'zone': ticket.track_section.start_station.division.zone.name,
            'issue': ticket.title,
            'team': ticket.assigned_team.team_name if ticket.assigned_team else 'Unassigned',
            'priority': ticket.priority,
            'status': ticket.status,
            'eta': (
                f"{int(ticket.estimated_duration_hours)} Hours"
                if ticket.estimated_duration_hours
                else 'TBD'
            ),
        })

    context = {
        'page_title': 'Maintenance Tickets',
        'tickets': filtered_tickets,
        'summary': summary,
        'current_filter': status_filter,
    }
    return render(request, 'tickets.html', context)
