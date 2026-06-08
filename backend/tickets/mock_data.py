# backend/tickets/mock_data.py
"""
Hardcoded mock maintenance ticket data for Phase 1.

Simulates the maintenance workflow with Indian engineer names,
realistic repair timelines, and Indian Railways terminology.
"""

TICKETS = [
    {
        'id': 'TKT-001',
        'linked_alert': 'ALT-2026-001',
        'priority': 'critical',
        'status': 'assigned',
        'team': 'Eastern Maintenance Unit',
        'eta': '4 Hours',
        'track_id': 'TRK-HWH-004',
        'section': 'Howrah — Bandel Junction',
        'issue': 'Rail Fracture — Emergency Weld Repair',
    },
    {
        'id': 'TKT-002',
        'linked_alert': 'ALT-2026-002',
        'priority': 'high',
        'status': 'scheduled',
        'team': 'Central Railway Team',
        'eta': '12 Hours',
        'track_id': 'TRK-MUM-002',
        'section': 'Mumbai CST — Thane',
        'issue': 'Rail Temperature Monitoring',
    },
    {
        'id': 'TKT-003',
        'linked_alert': 'ALT-2026-003',
        'priority': 'medium',
        'status': 'assigned',
        'team': 'Northern Maintenance Unit',
        'eta': '24 Hours',
        'track_id': 'TRK-NDL-001',
        'section': 'New Delhi — Ghaziabad',
        'issue': 'Track Geometry Correction',
    },
    {
        'id': 'TKT-004',
        'linked_alert': 'ALT-2026-004',
        'priority': 'low',
        'status': 'scheduled',
        'team': 'Southern Track Gang',
        'eta': '48 Hours',
        'track_id': 'TRK-MAS-003',
        'section': 'Chennai Central — Tambaram',
        'issue': 'Scheduled Ultrasonic Flaw Detection',
    },
    {
        'id': 'TKT-005',
        'linked_alert': 'ALT-2026-005',
        'priority': 'critical',
        'status': 'assigned',
        'team': 'Western Emergency Unit',
        'eta': '2 Hours',
        'track_id': 'TRK-ADI-009',
        'section': 'Ahmedabad Junction — Surat',
        'issue': 'Signal Failure Investigation',
    }
]

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
