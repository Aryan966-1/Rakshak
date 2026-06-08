# backend/tickets/mock_data.py
"""
Hardcoded mock maintenance ticket data for Phase 1.

Simulates the maintenance workflow with Indian engineer names,
realistic repair timelines, and Indian Railways terminology.
"""

TICKETS = [
    {
        'id': 'TKT-2026-001',
        'track_id': 'TRK-HWH-004',
        'section': 'Howrah — Bandel Junction',
        'issue': 'Rail Fracture — Emergency Weld Repair',
        'description': 'Micro-fracture detected by ultrasonic sensor at km 18.7. '
                       'Emergency thermit welding required. Speed restriction 30 km/h applied.',
        'assigned_to': 'Rajesh Kumar',
        'designation': 'Senior Section Engineer (P.Way)',
        'priority': 'high',
        'status': 'in_progress',
        'eta': '4 hours',
        'created': '2026-06-07 14:30:00',
        'gang': 'Gang 7 — Bandel',
    },
    {
        'id': 'TKT-2026-002',
        'track_id': 'TRK-MUM-002',
        'section': 'Mumbai CST — Thane',
        'issue': 'Rail Temperature Monitoring — Heat Patrol',
        'description': 'Deploy heat patrol gang for continuous monitoring. '
                       'Pre-position destressing equipment at km 34.',
        'assigned_to': 'Priya Sharma',
        'designation': 'Assistant Engineer (Track)',
        'priority': 'high',
        'status': 'in_progress',
        'eta': '6 hours',
        'created': '2026-06-07 13:50:00',
        'gang': 'Gang 12 — Kalyan',
    },
    {
        'id': 'TKT-2026-003',
        'track_id': 'TRK-SEC-008',
        'section': 'Secunderabad — Kazipet',
        'issue': 'Track Geometry Correction — Gauge Restoration',
        'description': 'Gauge deviation +3.2mm at km 55. Deploy tamping machine '
                       'and restore geometry to BG standard 1676mm.',
        'assigned_to': 'Amit Patel',
        'designation': 'Section Engineer (P.Way)',
        'priority': 'medium',
        'status': 'open',
        'eta': '12 hours',
        'created': '2026-06-07 12:20:00',
        'gang': 'Gang 3 — Kazipet',
    },
    {
        'id': 'TKT-2026-004',
        'track_id': 'TRK-JPR-006',
        'section': 'Jaipur — Ajmer',
        'issue': 'Turnout Switch Rail Replacement',
        'description': 'Switch rail worn beyond permissible limits at Jaipur yard '
                       'Turnout No. 14A. Complete replacement during traffic block.',
        'assigned_to': 'Sunita Rao',
        'designation': 'Senior Section Engineer (Points & Crossings)',
        'priority': 'medium',
        'status': 'open',
        'eta': '24 hours',
        'created': '2026-06-07 10:30:00',
        'gang': 'Gang 9 — Jaipur',
    },
    {
        'id': 'TKT-2026-005',
        'track_id': 'TRK-HWH-004',
        'section': 'Howrah — Bandel Junction',
        'issue': 'Ballast Replenishment — Post-Monsoon',
        'description': 'Ballast erosion detected on embankment section near Howrah. '
                       'Deploy ballast train and replenish 200m stretch.',
        'assigned_to': 'Vikram Singh',
        'designation': 'Assistant Engineer (Bridges & Earthwork)',
        'priority': 'high',
        'status': 'open',
        'eta': '8 hours',
        'created': '2026-06-07 08:45:00',
        'gang': 'Gang 7 — Bandel',
    },
    {
        'id': 'TKT-2026-006',
        'track_id': 'TRK-LKO-007',
        'section': 'Lucknow NR — Kanpur Central',
        'issue': 'Scheduled Ultrasonic Flaw Detection',
        'description': 'Routine USFD testing overdue by 24 hours. Schedule testing car '
                       'run during non-peak window (01:00–04:00).',
        'assigned_to': 'Deepak Verma',
        'designation': 'Section Engineer (Testing)',
        'priority': 'low',
        'status': 'open',
        'eta': '36 hours',
        'created': '2026-06-07 09:15:00',
        'gang': 'USFD Unit — Lucknow',
    },
    {
        'id': 'TKT-2026-007',
        'track_id': 'TRK-NDL-001',
        'section': 'New Delhi — Ghaziabad',
        'issue': 'Signal Cable Repair — Track Circuit',
        'description': 'Damaged track-circuit cable at km 12.4 near Ghaziabad. '
                       'S&T department coordinating joint repair.',
        'assigned_to': 'Meera Nair',
        'designation': 'Section Engineer (Signal)',
        'priority': 'high',
        'status': 'resolved',
        'eta': 'Completed',
        'created': '2026-06-06 20:45:00',
        'gang': 'S&T Gang — Ghaziabad',
    },
    {
        'id': 'TKT-2026-008',
        'track_id': 'TRK-SEC-008',
        'section': 'Secunderabad — Kazipet',
        'issue': 'Weld Joint Stress — Destressing Required',
        'description': 'Thermit weld at km 42.3 showing 15% stress elevation. '
                       'Schedule CWR destressing during next traffic block.',
        'assigned_to': 'Amit Patel',
        'designation': 'Section Engineer (P.Way)',
        'priority': 'medium',
        'status': 'open',
        'eta': '48 hours',
        'created': '2026-06-07 06:30:00',
        'gang': 'Gang 3 — Kazipet',
    },
    {
        'id': 'TKT-2026-009',
        'track_id': 'TRK-JPR-006',
        'section': 'Jaipur — Ajmer',
        'issue': 'Fishplate Bolt Retorquing',
        'description': 'IoT torque sensors flagged loose fishplate bolts at km 87.6. '
                       'Retorque all bolts in 500m radius per RDSO spec.',
        'assigned_to': 'Sunita Rao',
        'designation': 'Senior Section Engineer (Points & Crossings)',
        'priority': 'low',
        'status': 'open',
        'eta': '72 hours',
        'created': '2026-06-06 18:30:00',
        'gang': 'Gang 9 — Jaipur',
    },
    {
        'id': 'TKT-2026-010',
        'track_id': 'TRK-BLR-005',
        'section': 'Bengaluru City — Whitefield',
        'issue': 'Post-Tamping Verification',
        'description': 'Track realignment and tamping completed. Verify geometry '
                       'parameters with track recording car run.',
        'assigned_to': 'Kavitha Reddy',
        'designation': 'Assistant Engineer (Track Machine)',
        'priority': 'low',
        'status': 'resolved',
        'eta': 'Completed',
        'created': '2026-06-06 22:15:00',
        'gang': 'Gang 14 — Whitefield',
    },
]

# ---------------------------------------------------------------------------
# Summary counts
# ---------------------------------------------------------------------------
TICKET_SUMMARY = {
    'total': len(TICKETS),
    'open': len([t for t in TICKETS if t['status'] == 'open']),
    'in_progress': len([t for t in TICKETS if t['status'] == 'in_progress']),
    'resolved': len([t for t in TICKETS if t['status'] == 'resolved']),
    'high_priority': len([t for t in TICKETS if t['priority'] == 'high']),
}
