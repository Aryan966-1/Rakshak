# backend/map_view/mock_data.py
"""
Hardcoded mock railway station and map data for Phase 1.

Contains real GPS coordinates for major Indian Railway stations
along with simulated alert markers for the Leaflet map view.
"""

# ---------------------------------------------------------------------------
# Railway stations with real coordinates
# ---------------------------------------------------------------------------
STATIONS = [
    {
        'name': 'New Delhi',
        'code': 'NDLS',
        'lat': 28.6139,
        'lng': 77.2090,
        'zone': 'Northern Railway',
        'tracks_monitored': 24,
        'status': 'warning',
        'active_alerts': 1,
        'daily_trains': 350,
    },
    {
        'name': 'Mumbai CST',
        'code': 'CSMT',
        'lat': 18.9398,
        'lng': 72.8355,
        'zone': 'Central Railway',
        'tracks_monitored': 18,
        'status': 'critical',
        'active_alerts': 3,
        'daily_trains': 420,
    },
    {
        'name': 'Chennai Central',
        'code': 'MAS',
        'lat': 13.0827,
        'lng': 80.2707,
        'zone': 'Southern Railway',
        'tracks_monitored': 16,
        'status': 'healthy',
        'active_alerts': 0,
        'daily_trains': 280,
    },
    {
        'name': 'Howrah Junction',
        'code': 'HWH',
        'lat': 22.5839,
        'lng': 88.3428,
        'zone': 'Eastern Railway',
        'tracks_monitored': 22,
        'status': 'critical',
        'active_alerts': 4,
        'daily_trains': 380,
    },
    {
        'name': 'Bengaluru City',
        'code': 'SBC',
        'lat': 12.9784,
        'lng': 77.5712,
        'zone': 'South Western Railway',
        'tracks_monitored': 12,
        'status': 'healthy',
        'active_alerts': 0,
        'daily_trains': 190,
    },
    {
        'name': 'Jaipur Junction',
        'code': 'JP',
        'lat': 26.9194,
        'lng': 75.7880,
        'zone': 'North Western Railway',
        'tracks_monitored': 10,
        'status': 'warning',
        'active_alerts': 2,
        'daily_trains': 145,
    },
    {
        'name': 'Lucknow NR',
        'code': 'LKO',
        'lat': 26.8467,
        'lng': 80.9462,
        'zone': 'Northern Railway',
        'tracks_monitored': 14,
        'status': 'healthy',
        'active_alerts': 0,
        'daily_trains': 210,
    },
    {
        'name': 'Secunderabad Junction',
        'code': 'SC',
        'lat': 17.4344,
        'lng': 78.5013,
        'zone': 'South Central Railway',
        'tracks_monitored': 16,
        'status': 'warning',
        'active_alerts': 2,
        'daily_trains': 230,
    },
    {
        'name': 'Ahmedabad Junction',
        'code': 'ADI',
        'lat': 23.0225,
        'lng': 72.5714,
        'zone': 'Western Railway',
        'tracks_monitored': 12,
        'status': 'healthy',
        'active_alerts': 0,
        'daily_trains': 175,
    },
    {
        'name': 'Patna Junction',
        'code': 'PNBE',
        'lat': 25.6093,
        'lng': 85.1376,
        'zone': 'East Central Railway',
        'tracks_monitored': 10,
        'status': 'healthy',
        'active_alerts': 1,
        'daily_trains': 160,
    },
    {
        'name': 'Bhopal Junction',
        'code': 'BPL',
        'lat': 23.2689,
        'lng': 77.4124,
        'zone': 'West Central Railway',
        'tracks_monitored': 8,
        'status': 'healthy',
        'active_alerts': 0,
        'daily_trains': 135,
    },
    {
        'name': 'Guwahati',
        'code': 'GHY',
        'lat': 26.1445,
        'lng': 91.7362,
        'zone': 'Northeast Frontier Railway',
        'tracks_monitored': 6,
        'status': 'warning',
        'active_alerts': 1,
        'daily_trains': 85,
    },
]

# ---------------------------------------------------------------------------
# Railway route lines connecting major stations (for polyline rendering)
# ---------------------------------------------------------------------------
RAIL_ROUTES = [
    {
        'name': 'Delhi — Mumbai Rajdhani Route',
        'train': '12951 Mumbai Rajdhani',
        'coordinates': [
            [28.6139, 77.2090],   # New Delhi
            [26.9194, 75.7880],   # Jaipur
            [23.2689, 77.4124],   # Bhopal
            [18.9398, 72.8355],   # Mumbai CST
        ],
        'status': 'warning',
    },
    {
        'name': 'Delhi — Howrah Rajdhani Route',
        'train': '12301 Rajdhani Express',
        'coordinates': [
            [28.6139, 77.2090],   # New Delhi
            [26.8467, 80.9462],   # Lucknow
            [25.6093, 85.1376],   # Patna
            [22.5839, 88.3428],   # Howrah
        ],
        'status': 'critical',
    },
    {
        'name': 'Chennai — Bengaluru Shatabdi Route',
        'train': '12027 Chennai Shatabdi',
        'coordinates': [
            [13.0827, 80.2707],   # Chennai
            [12.9784, 77.5712],   # Bengaluru
        ],
        'status': 'healthy',
    },
    {
        'name': 'Secunderabad — Chennai Route',
        'train': '12759 Charminar Express',
        'coordinates': [
            [17.4344, 78.5013],   # Secunderabad
            [13.0827, 80.2707],   # Chennai
        ],
        'status': 'warning',
    },
]
