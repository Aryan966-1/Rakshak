# backend/alerts/mock_data.py
"""
Hardcoded mock alert data for Phase 1.

Simulates realistic railway infrastructure alerts at varying
severity levels across Indian Railway zones.
"""

ALERTS = [
    {
        'id': 'ALT-2026-001',
        'severity': 'critical',
        'title': 'Rail Fracture Detected',
        'description': 'Ultrasonic sensor detected micro-fracture on rail head. '
                       'Immediate speed restriction recommended on Track TRK-HWH-004.',
        'track_id': 'TRK-HWH-004',
        'section': 'Howrah — Bandel Junction',
        'station': 'Bandel Junction',
        'zone': 'Eastern Railway',
        'timestamp': '2026-06-07 14:23:00',
        'status': 'active',
    },
    {
        'id': 'ALT-2026-002',
        'severity': 'critical',
        'title': 'Excessive Rail Temperature',
        'description': 'Rail temperature exceeding 52°C on Mumbai suburban corridor. '
                       'Risk of rail buckling. Speed restriction applied.',
        'track_id': 'TRK-MUM-002',
        'section': 'Mumbai CST — Thane',
        'station': 'Kalyan Junction',
        'zone': 'Central Railway',
        'timestamp': '2026-06-07 13:45:00',
        'status': 'active',
    },
    {
        'id': 'ALT-2026-003',
        'severity': 'warning',
        'title': 'Gauge Deviation Warning',
        'description': 'Rail gauge deviation of +3.2mm detected. Track geometry '
                       'degrading — maintenance recommended within 48 hours.',
        'track_id': 'TRK-SEC-008',
        'section': 'Secunderabad — Kazipet',
        'station': 'Warangal',
        'zone': 'South Central Railway',
        'timestamp': '2026-06-07 12:10:00',
        'status': 'active',
    },
    {
        'id': 'ALT-2026-004',
        'severity': 'warning',
        'title': 'Vibration Anomaly',
        'description': 'Elevated vibration levels (4.8 mm/s) on Mumbai section. '
                       'Possible ballast settlement or loose fastener.',
        'track_id': 'TRK-MUM-002',
        'section': 'Mumbai CST — Thane',
        'station': 'Dombivli',
        'zone': 'Central Railway',
        'timestamp': '2026-06-07 11:30:00',
        'status': 'active',
    },
    {
        'id': 'ALT-2026-005',
        'severity': 'warning',
        'title': 'Turnout Wear Detected',
        'description': 'Switch rail wear exceeding threshold at Jaipur yard. '
                       'Replacement scheduling advised.',
        'track_id': 'TRK-JPR-006',
        'section': 'Jaipur — Ajmer',
        'station': 'Jaipur Junction',
        'zone': 'North Western Railway',
        'timestamp': '2026-06-07 10:15:00',
        'status': 'active',
    },
    {
        'id': 'ALT-2026-006',
        'severity': 'info',
        'title': 'Scheduled Inspection Due',
        'description': 'Routine ultrasonic flaw detection due on Lucknow–Kanpur section. '
                       'Last inspection: 48 hours ago.',
        'track_id': 'TRK-LKO-007',
        'section': 'Lucknow NR — Kanpur Central',
        'station': 'Lucknow NR',
        'zone': 'Northern Railway',
        'timestamp': '2026-06-07 09:00:00',
        'status': 'active',
    },
    {
        'id': 'ALT-2026-007',
        'severity': 'critical',
        'title': 'Ballast Washout Risk',
        'description': 'Heavy rainfall detected near Howrah. Ballast erosion sensors '
                       'triggered on embankment section. Flood alert active.',
        'track_id': 'TRK-HWH-004',
        'section': 'Howrah — Bandel Junction',
        'station': 'Howrah',
        'zone': 'Eastern Railway',
        'timestamp': '2026-06-07 08:30:00',
        'status': 'active',
    },
    {
        'id': 'ALT-2026-008',
        'severity': 'info',
        'title': 'Sensor Calibration Complete',
        'description': 'All vibration sensors on Chennai section recalibrated successfully. '
                       'Readings verified within ±0.1 mm/s tolerance.',
        'track_id': 'TRK-CHN-003',
        'section': 'Chennai Central — Tambaram',
        'station': 'Chennai Central',
        'zone': 'Southern Railway',
        'timestamp': '2026-06-07 07:45:00',
        'status': 'resolved',
    },
    {
        'id': 'ALT-2026-009',
        'severity': 'warning',
        'title': 'Weld Joint Stress Elevated',
        'description': 'Thermit weld joint showing stress concentration at km 42.3. '
                       'Strain gauge reading 15% above baseline.',
        'track_id': 'TRK-SEC-008',
        'section': 'Secunderabad — Kazipet',
        'station': 'Secunderabad',
        'zone': 'South Central Railway',
        'timestamp': '2026-06-07 06:20:00',
        'status': 'active',
    },
    {
        'id': 'ALT-2026-010',
        'severity': 'info',
        'title': 'Track Realignment Complete',
        'description': 'Bengaluru section track realignment completed by Gang 14. '
                       'Post-tamping readings nominal.',
        'track_id': 'TRK-BLR-005',
        'section': 'Bengaluru City — Whitefield',
        'station': 'Bengaluru City',
        'zone': 'South Western Railway',
        'timestamp': '2026-06-06 22:00:00',
        'status': 'resolved',
    },
    {
        'id': 'ALT-2026-011',
        'severity': 'critical',
        'title': 'Signal Cable Damage',
        'description': 'Track-circuit signal cable damaged near Ghaziabad. '
                       'Manual caution order issued for Down line.',
        'track_id': 'TRK-NDL-001',
        'section': 'New Delhi — Ghaziabad',
        'station': 'Ghaziabad',
        'zone': 'Northern Railway',
        'timestamp': '2026-06-06 20:30:00',
        'status': 'resolved',
    },
    {
        'id': 'ALT-2026-012',
        'severity': 'warning',
        'title': 'Fishplate Bolt Loosening',
        'description': 'IoT torque sensor detected loosening fishplate bolts at '
                       'km 87.6 on Jaipur–Ajmer section.',
        'track_id': 'TRK-JPR-006',
        'section': 'Jaipur — Ajmer',
        'station': 'Ajmer Junction',
        'zone': 'North Western Railway',
        'timestamp': '2026-06-06 18:15:00',
        'status': 'active',
    },
]

# ---------------------------------------------------------------------------
# Summary counts derived from the alert list
# ---------------------------------------------------------------------------
ALERT_SUMMARY = {
    'total': len(ALERTS),
    'critical': len([a for a in ALERTS if a['severity'] == 'critical']),
    'warning': len([a for a in ALERTS if a['severity'] == 'warning']),
    'info': len([a for a in ALERTS if a['severity'] == 'info']),
    'active': len([a for a in ALERTS if a['status'] == 'active']),
    'resolved': len([a for a in ALERTS if a['status'] == 'resolved']),
}
