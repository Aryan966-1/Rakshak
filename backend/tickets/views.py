# backend/tickets/views.py
"""
Tickets view — displays maintenance tickets.

Renders the ticket management page showing assigned engineers,
priorities, status, and estimated repair times.
"""

from django.shortcuts import render

from .mock_data import TICKETS, TICKET_SUMMARY


def tickets_page(request):
    """Render the maintenance tickets page."""
    # Optional status filter from query string (e.g., ?status=open)
    status_filter = request.GET.get('status', 'all')

    if status_filter != 'all':
        filtered_tickets = [t for t in TICKETS if t['status'] == status_filter]
    else:
        filtered_tickets = TICKETS

    context = {
        'page_title': 'Maintenance Tickets',
        'tickets': filtered_tickets,
        'summary': TICKET_SUMMARY,
        'current_filter': status_filter,
    }
    return render(request, 'tickets.html', context)
