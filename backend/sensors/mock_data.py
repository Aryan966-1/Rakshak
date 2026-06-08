# backend/sensors/mock_data.py
"""
Hardcoded mock sensor and track health data for Phase 1.

All data simulates realistic Indian Railways telemetry readings.
In future phases this will be replaced by live database queries
and real-time sensor ingestion pipelines.
"""

from alerts.mock_data import ALERTS
from map_view.mock_data import STATIONS



# ---------------------------------------------------------------------------
# Sensor trend data — 24-hour readings for Chart.js line charts
# ---------------------------------------------------------------------------
SENSOR_TRENDS = {
    'timestamps': [
        '00:00', '04:00', '08:00', '12:00', '16:00', '20:00', '24:00'
    ],
    # Vibration amplitude (mm/s)
    'vibration': [2.1, 2.4, 2.8, 3.2, 3.7, 4.5, 5.8],
    # Rail temperature (°C)
    'temperature': [34, 35, 37, 39, 42, 45, 48],
    # Rail gauge deviation (mm from standard 1676mm)
    'gauge_deviation': [0.8, 1.2, 1.8, 2.1, 2.7, 3.0, 3.2],
}

# ---------------------------------------------------------------------------
# Track sections with health scores
# ---------------------------------------------------------------------------
TRACK_SECTIONS = [
    {
        'id': 'TRK-NDL-001',
        'section': 'New Delhi — Ghaziabad',
        'zone': 'Northern Railway',
        'health': 94,
        'status': 'healthy',
        'last_inspected': '2026-06-06',
        'trains_daily': 142,
    },
    {
        'id': 'TRK-MUM-002',
        'section': 'Mumbai CST — Thane',
        'zone': 'Central Railway',
        'health': 67,
        'status': 'warning',
        'last_inspected': '2026-06-05',
        'trains_daily': 218,
    },
    {
        'id': 'TRK-CHN-003',
        'section': 'Chennai Central — Tambaram',
        'zone': 'Southern Railway',
        'health': 91,
        'status': 'healthy',
        'last_inspected': '2026-06-07',
        'trains_daily': 164,
    },
    {
        'id': 'TRK-HWH-004',
        'section': 'Howrah — Bandel Junction',
        'zone': 'Eastern Railway',
        'health': 45,
        'status': 'critical',
        'last_inspected': '2026-06-03',
        'trains_daily': 196,
    },
    {
        'id': 'TRK-BLR-005',
        'section': 'Bengaluru City — Whitefield',
        'zone': 'South Western Railway',
        'health': 88,
        'status': 'healthy',
        'last_inspected': '2026-06-07',
        'trains_daily': 98,
    },
    {
        'id': 'TRK-JPR-006',
        'section': 'Jaipur — Ajmer',
        'zone': 'North Western Railway',
        'health': 72,
        'status': 'warning',
        'last_inspected': '2026-06-04',
        'trains_daily': 76,
    },
    {
        'id': 'TRK-LKO-007',
        'section': 'Lucknow NR — Kanpur Central',
        'zone': 'Northern Railway',
        'health': 83,
        'status': 'healthy',
        'last_inspected': '2026-06-06',
        'trains_daily': 134,
    },
    {
        'id': 'TRK-SEC-008',
        'section': 'Secunderabad — Kazipet',
        'zone': 'South Central Railway',
        'health': 58,
        'status': 'warning',
        'last_inspected': '2026-06-02',
        'trains_daily': 112,
    },
]

# ---------------------------------------------------------------------------
# Recent sensor readings (last 5 for dashboard mini-table)
# ---------------------------------------------------------------------------
RECENT_READINGS = [
    {'track_id': 'TRK-HWH-004', 'sensor': 'Vibration', 'value': '5.8 mm/s', 'status': 'critical', 'time': '14:23'},
    {'track_id': 'TRK-MUM-002', 'sensor': 'Temperature', 'value': '48°C', 'status': 'warning', 'time': '14:15'},
    {'track_id': 'TRK-SEC-008', 'sensor': 'Gauge', 'value': '+3.2 mm', 'status': 'warning', 'time': '14:08'},
    {'track_id': 'TRK-NDL-001', 'sensor': 'Vibration', 'value': '2.1 mm/s', 'status': 'healthy', 'time': '14:01'},
    {'track_id': 'TRK-CHN-003', 'sensor': 'Temperature', 'value': '34°C', 'status': 'healthy', 'time': '13:55'},
]

# ---------------------------------------------------------------------------
# KPI Summary — displayed on the dashboard header cards
# ---------------------------------------------------------------------------
overall_health = sum(t['health'] for t in TRACK_SECTIONS) / len(TRACK_SECTIONS)
active_alerts_count = len([a for a in ALERTS if a['status'] == 'active'])
predicted_failures_count = len([a for a in ALERTS if a['severity'] == 'critical' and a['status'] == 'active'])
tracks_monitored_count = sum(s['tracks_monitored'] for s in STATIONS)

KPI_SUMMARY = {
    'overall_health': round(overall_health, 1),
    'active_alerts': active_alerts_count,
    'predicted_failures': predicted_failures_count,
    'cost_savings': 24_50_000,    # ₹24.5 Lakhs saved via predictive maintenance
    'tracks_monitored': tracks_monitored_count,
}
