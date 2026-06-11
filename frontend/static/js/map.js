// frontend/static/js/map.js
//
// Rakshak — Railway Map Initialization (Database-Driven)
// Fetches data from Django API endpoints and renders:
//   - Station markers with color-coded status
//   - Rail route polylines
//   - Ticket markers with priority colors
//   - Alert markers with severity indicators
//   - Interactive popup cards with details

'use strict';

// ================================================================
// STATUS COLOR PALETTE — shared across all marker types
// ================================================================
var STATUS_COLORS = {
    healthy:  '#10b981',
    warning:  '#f59e0b',
    critical: '#ef4444',
};

var STATUS_GLOW = {
    healthy:  'rgba(16,185,129,0.4)',
    warning:  'rgba(245,158,11,0.4)',
    critical: 'rgba(239,68,68,0.5)',
};

// Ticket priority → marker color
var TICKET_COLORS = {
    critical: '#ef4444',   // Red
    high:     '#f59e0b',   // Amber
    medium:   '#3b82f6',   // Blue
    low:      '#10b981',   // Green
};

// Ticket status → marker color (overrides priority for active tickets)
var TICKET_STATUS_COLORS = {
    open:        '#3b82f6',   // Blue — Active
    assigned:    '#3b82f6',   // Blue — Active
    in_progress: '#f59e0b',   // Yellow — In progress
    scheduled:   '#f59e0b',   // Yellow — Scheduled
    resolved:    '#10b981',   // Green — Healthy
};

// Alert severity → marker color
var ALERT_SEVERITY_COLORS = {
    critical: '#ef4444',
    warning:  '#f59e0b',
    info:     '#3b82f6',
};


// ================================================================
// MAIN MAP INITIALIZATION
// ================================================================

/**
 * Initialize the Leaflet railway map by fetching data from APIs.
 * Called from map.html after DOM load.
 */
function initRailwayMapFromAPI() {
    var mapContainer = document.getElementById('railway-map');
    if (!mapContainer) return;

    // ----------------------------------------------------------------
    // Create map centered on India
    // ----------------------------------------------------------------
    var map = L.map('railway-map', {
        center: [22.5, 79.0],
        zoom: 5,
        minZoom: 4,
        maxZoom: 14,
        zoomControl: true,
    });

    // Dark tile layer — CartoDB Dark Matter for the operations center look
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
        subdomains: 'abcd',
        maxZoom: 20,
    }).addTo(map);

    // ----------------------------------------------------------------
    // Inject custom CSS for map elements
    // ----------------------------------------------------------------
    _injectMapStyles();

    // ----------------------------------------------------------------
    // Fetch all data in parallel, then render layers
    // ----------------------------------------------------------------
    Promise.all([
        fetch('/api/stations/').then(function(r) { return r.json(); }),
        fetch('/api/routes/').then(function(r) { return r.json(); }),
        fetch('/api/tickets/').then(function(r) { return r.json(); }),
        fetch('/api/alerts/').then(function(r) { return r.json(); }),
        fetch('/api/summary/').then(function(r) { return r.json(); }),
    ]).then(function(results) {
        var stations = results[0];
        var routes   = results[1];
        var tickets  = results[2];
        var alerts   = results[3];
        var summary  = results[4];

        console.log('Loaded:', stations.length, 'stations,', routes.length, 'routes,', tickets.length, 'tickets,', alerts.length, 'alerts');

        // Update stats bar with live DB counts
        _updateStatsBar(summary);

        // Render layers (order matters: routes first, then markers on top)
        _renderRoutes(map, routes);
        _renderStations(map, stations);
        _renderTickets(map, tickets);
        _renderAlerts(map, alerts);

        // Fit map to show entire Indian railway network
        var indiaBounds = L.latLngBounds(
            [6.5, 68.0],    // Southwest (Kanyakumari / Gujarat coast)
            [35.5, 97.5]    // Northeast (Kashmir / Arunachal Pradesh)
        );
        map.fitBounds(indiaBounds, { padding: [20, 20] });

        // Start train simulation (Phase 8)
        if (typeof initTrainSimulation === 'function') {
            initTrainSimulation(map);
        }

    }).catch(function(err) {
        console.error('Failed to load map data:', err);
    });
}


// ================================================================
// STATS BAR UPDATE — populate from API summary
// ================================================================
function _updateStatsBar(summary) {
    var statValues = document.querySelectorAll('.map-stat-value');
    if (statValues.length >= 4) {
        statValues[0].textContent = summary.stations || '—';
        statValues[1].textContent = summary.track_sections || '—';
        statValues[2].textContent = summary.active_routes || '—';
        statValues[3].textContent = summary.railway_zones || '—';
    }
}


// ================================================================
// STATION MARKERS
// ================================================================
function _renderStations(map, stations) {
    stations.forEach(function(station) {
        var color = STATUS_COLORS[station.status] || STATUS_COLORS.healthy;
        var glow  = STATUS_GLOW[station.status]  || STATUS_GLOW.healthy;

        // Outer glow circle
        L.circleMarker([station.lat, station.lng], {
            radius: 16,
            color: 'transparent',
            fillColor: glow,
            fillOpacity: 0.3,
            interactive: false,
        }).addTo(map);

        // Inner station marker
        var marker = L.circleMarker([station.lat, station.lng], {
            radius: 8,
            color: color,
            fillColor: color,
            fillOpacity: 0.9,
            weight: 2,
        }).addTo(map);

        // Build popup HTML
        var statusLabel = station.status.charAt(0).toUpperCase() + station.status.slice(1);
        var alertsText = station.active_alerts > 0
            ? '<span style="color:' + STATUS_COLORS.critical + '">' + station.active_alerts + '</span>'
            : '<span style="color:' + STATUS_COLORS.healthy + '">0</span>';

        var popupHtml = '<div class="station-popup">' +
            '<h4>' + station.name + '</h4>' +
            '<div class="popup-code">Station Code: ' + station.code + '</div>' +
            '<div class="popup-detail"><span class="popup-label">Zone</span><span class="popup-value">' + station.zone + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Division</span><span class="popup-value">' + station.division + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Status</span><span class="popup-value" style="color:' + color + '">' + statusLabel + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Tracks Monitored</span><span class="popup-value">' + station.tracks_monitored + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Active Alerts</span><span class="popup-value">' + alertsText + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Daily Trains</span><span class="popup-value">' + station.daily_trains + '</span></div>' +
            '</div>';

        marker.bindPopup(popupHtml, {
            maxWidth: 280,
            className: 'dark-popup',
        });

        // Station code label
        marker.bindTooltip(station.code, {
            permanent: true,
            direction: 'top',
            offset: [0, -12],
            className: 'station-label',
        });
    });
}


// ================================================================
// ROUTE POLYLINES
// ================================================================
function _renderRoutes(map, routes) {
    routes.forEach(function(route) {
        var color = STATUS_COLORS[route.status] || '#3b82f6';

        var polyline = L.polyline(route.coordinates, {
            color: color,
            weight: 2.5,
            opacity: 0.6,
            dashArray: '8, 6',
            lineJoin: 'round',
        }).addTo(map);

        // Route tooltip on hover
        var distText = route.distance_km ? route.distance_km + ' km' : '';
        polyline.bindTooltip(
            '<strong>' + route.name + '</strong><br>' +
            '<span style="font-size:0.8em;color:#94a3b8">' + route.train + '</span>' +
            (distText ? '<br><span style="font-size:0.75em;color:#64748b">' + distText + '</span>' : ''),
            { sticky: true }
        );
    });
}


// ================================================================
// TICKET MARKERS — color-coded by priority/status
// ================================================================
function _renderTickets(map, tickets) {
    tickets.forEach(function(ticket) {
        // Use status-based color, falling back to priority-based
        var color = TICKET_STATUS_COLORS[ticket.status] || TICKET_COLORS[ticket.priority] || '#3b82f6';

        var marker = L.circleMarker([ticket.lat, ticket.lng], {
            radius: 5,
            color: color,
            fillColor: color,
            fillOpacity: 0.7,
            weight: 1.5,
        }).addTo(map);

        var priorityLabel = ticket.priority.charAt(0).toUpperCase() + ticket.priority.slice(1);
        var statusLabel = ticket.status.replace(/_/g, ' ');
        statusLabel = statusLabel.charAt(0).toUpperCase() + statusLabel.slice(1);

        var popupHtml = '<div class="station-popup">' +
            '<h4>🎫 ' + ticket.id + '</h4>' +
            '<div class="popup-code">' + ticket.title + '</div>' +
            '<div class="popup-detail"><span class="popup-label">Zone</span><span class="popup-value">' + ticket.zone + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Division</span><span class="popup-value">' + ticket.division + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Station</span><span class="popup-value">' + ticket.station + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Status</span><span class="popup-value" style="color:' + color + '">' + statusLabel + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Priority</span><span class="popup-value" style="color:' + (TICKET_COLORS[ticket.priority] || '#fff') + '">' + priorityLabel + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Team</span><span class="popup-value">' + ticket.team + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Section</span><span class="popup-value">' + ticket.section + '</span></div>' +
            '</div>';

        marker.bindPopup(popupHtml, {
            maxWidth: 300,
            className: 'dark-popup',
        });
    });
}


// ================================================================
// ALERT MARKERS — pulsing severity indicators
// ================================================================
function _renderAlerts(map, alerts) {
    alerts.forEach(function(alert) {
        var color = ALERT_SEVERITY_COLORS[alert.severity] || '#f59e0b';

        // Outer pulsing glow
        L.circleMarker([alert.lat, alert.lng], {
            radius: 20,
            color: 'transparent',
            fillColor: color,
            fillOpacity: 0.15,
            interactive: false,
        }).addTo(map);

        // Alert marker (diamond-shaped via DivIcon)
        var marker = L.circleMarker([alert.lat, alert.lng], {
            radius: 6,
            color: '#111827',
            fillColor: color,
            fillOpacity: 1.0,
            weight: 2,
        }).addTo(map);

        var popupHtml = '<div class="station-popup">' +
            '<h4>⚠️ ' + alert.title + '</h4>' +
            '<div class="popup-code">' + alert.id + '</div>' +
            '<div class="popup-detail"><span class="popup-label">Severity</span><span class="popup-value" style="color:' + color + '">' + alert.severity.toUpperCase() + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Zone</span><span class="popup-value">' + alert.zone + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Station</span><span class="popup-value">' + alert.station + '</span></div>' +
            '<div style="margin-top:8px;font-size:0.8em;color:#94a3b8">' + alert.description + '</div>' +
            '</div>';

        marker.bindPopup(popupHtml, {
            maxWidth: 320,
            className: 'dark-popup',
        });
    });
}


// ================================================================
// INJECT MAP STYLES — station labels, popups, tooltips
// ================================================================
function _injectMapStyles() {
    var style = document.createElement('style');
    style.textContent =
        '.station-label {' +
        '  background: rgba(17, 24, 39, 0.9) !important;' +
        '  color: #f1f5f9 !important;' +
        '  border: 1px solid rgba(255,255,255,0.1) !important;' +
        '  font-family: "JetBrains Mono", monospace !important;' +
        '  font-size: 10px !important;' +
        '  font-weight: 600 !important;' +
        '  letter-spacing: 1px !important;' +
        '  padding: 2px 6px !important;' +
        '  border-radius: 4px !important;' +
        '  box-shadow: 0 2px 8px rgba(0,0,0,0.3) !important;' +
        '}' +
        '.station-label::before { border-top-color: rgba(17,24,39,0.9) !important; }' +
        '.dark-popup .leaflet-popup-content-wrapper {' +
        '  background: #111827 !important;' +
        '  border: 1px solid rgba(255,255,255,0.06) !important;' +
        '}' +
        '.dark-popup .leaflet-popup-tip {' +
        '  background: #111827 !important;' +
        '}' +
        '.station-popup h4 {' +
        '  margin: 0 0 8px 0;' +
        '  color: #f1f5f9;' +
        '  font-size: 0.95rem;' +
        '  font-weight: 600;' +
        '}' +
        '.popup-code {' +
        '  font-family: "JetBrains Mono", monospace;' +
        '  font-size: 0.75rem;' +
        '  color: #64748b;' +
        '  margin-bottom: 8px;' +
        '}' +
        '.popup-detail {' +
        '  display: flex;' +
        '  justify-content: space-between;' +
        '  padding: 3px 0;' +
        '  border-bottom: 1px solid rgba(255,255,255,0.04);' +
        '}' +
        '.popup-label {' +
        '  color: #94a3b8;' +
        '  font-size: 0.8rem;' +
        '}' +
        '.popup-value {' +
        '  color: #f1f5f9;' +
        '  font-family: "JetBrains Mono", monospace;' +
        '  font-size: 0.8rem;' +
        '  font-weight: 500;' +
        '}';
    document.head.appendChild(style);
}
