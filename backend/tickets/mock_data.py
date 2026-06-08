# backend/tickets/mock_data.py
"""
Hardcoded mock maintenance ticket data for Phase 1.

Simulates the maintenance workflow with Indian engineer names,
realistic repair timelines, and Indian Railways terminology.
"""

from alerts.mock_data import ALERTS

_TICKETS_BASE = [
    {
        'id': 'TKT-001',
        'linked_alert': 'ALT-2026-001',
        'priority': 'critical',
        'status': 'assigned',
        'team': 'Eastern Maintenance Unit',
        'eta': '4 Hours',
        'issue': 'Rail Fracture — Emergency Weld Repair',
    },
    {
        'id': 'TKT-002',
        'linked_alert': 'ALT-2026-002',
        'priority': 'high',
        'status': 'scheduled',
        'team': 'Central Railway Team',
        'eta': '12 Hours',
        'issue': 'Rail Temperature Monitoring',
    },
    {
        'id': 'TKT-003',
        'linked_alert': 'ALT-2026-003',
        'priority': 'medium',
        'status': 'assigned',
        'team': 'Northern Maintenance Unit',
        'eta': '24 Hours',
        'issue': 'Track Geometry Correction',
    },
    {
        'id': 'TKT-004',
        'linked_alert': 'ALT-2026-004',
        'priority': 'low',
        'status': 'scheduled',
        'team': 'Southern Track Gang',
        'eta': '48 Hours',
        'issue': 'Scheduled Ultrasonic Flaw Detection',
    },
    {
        'id': 'TKT-005',
        'linked_alert': 'ALT-2026-005',
        'priority': 'critical',
        'status': 'assigned',
        'team': 'Western Emergency Unit',
        'eta': '2 Hours',
        'issue': 'Signal Failure Investigation',
    }
]

# Create an alert lookup dict for O(1) access
alert_lookup = {a['id']: a for a in ALERTS}

TICKETS = []
for tb in _TICKETS_BASE:
    linked_alert_id = tb.get('linked_alert')
    alert_info = alert_lookup.get(linked_alert_id)
    
    # Enrich with alert data if found, overriding any mismatched values
    if alert_info:
        tb['track_id'] = alert_info['track_id']
        tb['section'] = alert_info['section']
        tb['station'] = alert_info['station']
        tb['zone'] = alert_info['zone']
    else:
        tb['track_id'] = ''
        tb['section'] = ''
        tb['station'] = ''
        tb['zone'] = ''

    TICKETS.append(tb)

# ---------------------------------------------------------------------------
# Summary counts
# ---------------------------------------------------------------------------
TICKET_SUMMARY = {
    'total': len(TICKETS),
    'assigned': len([t for t in TICKETS if t['status'] == 'assigned']),
    'scheduled': len([t for t in TICKETS if t['status'] == 'scheduled']),
    'resolved': len([t for t in TICKETS if t['status'] == 'resolved']),
    'critical_priority': len([t for t in TICKETS if t['priority'] == 'critical']),
}
