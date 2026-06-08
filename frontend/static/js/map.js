// frontend/static/js/map.js
//
// Rakshak — Railway Map Initialization
// Creates a Leaflet map centered on India with:
//   - Dark tile layer (CartoDB Dark Matter)
//   - Station markers with color-coded status
//   - Rail route polylines
//   - Interactive popup cards with station details

'use strict';

/**
 * Initialize the Leaflet railway map.
 * Called from map.html after DOM load with parsed JSON data.
 *
 * @param {Array} stations - Array of station objects with lat/lng, name, status, etc.
 * @param {Array} routes - Array of route objects with coordinates, name, status.
 */

function initRailwayMap(stations, routes) {
    console.log("Stations:", stations);
    console.log("Routes:", routes);
    console.log("Station count:", stations.length);
    console.log("Route count:", routes.length);

    const mapContainer = document.getElementById('railway-map');
    if (!mapContainer) return;

    // ----------------------------------------------------------------
    // Create map centered on India
    // ----------------------------------------------------------------
    const map = L.map('railway-map', {
        center: [22.5, 79.0],  // Center of India
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
    // Color mapping for status levels
    // ----------------------------------------------------------------
    const statusColors = {
        healthy: '#10b981',
        warning: '#f59e0b',
        critical: '#ef4444',
    };

    const statusGlow = {
        healthy: 'rgba(16,185,129,0.4)',
        warning: 'rgba(245,158,11,0.4)',
        critical: 'rgba(239,68,68,0.5)',
    };

    // ----------------------------------------------------------------
    // Custom circle markers for stations
    // ----------------------------------------------------------------
    stations.forEach(function (station) {
        const color = statusColors[station.status] || statusColors.healthy;
        const glow = statusGlow[station.status] || statusGlow.healthy;

        // Create a circle marker with glow effect (outer + inner)
        // Outer glow circle
        L.circleMarker([station.lat, station.lng], {
            radius: 16,
            color: 'transparent',
            fillColor: glow,
            fillOpacity: 0.3,
            interactive: false,
        }).addTo(map);

        // Inner station marker
        const marker = L.circleMarker([station.lat, station.lng], {
            radius: 8,
            color: color,
            fillColor: color,
            fillOpacity: 0.9,
            weight: 2,
        }).addTo(map);

        // Build popup HTML
        var statusLabel = station.status.charAt(0).toUpperCase() + station.status.slice(1);
        var alertsText = station.active_alerts > 0
            ? '<span style="color:' + statusColors.critical + '">' + station.active_alerts + '</span>'
            : '<span style="color:' + statusColors.healthy + '">0</span>';

        var popupHtml = '<div class="station-popup">' +
            '<h4>' + station.name + '</h4>' +
            '<div class="popup-code">Station Code: ' + station.code + '</div>' +
            '<div class="popup-detail"><span class="popup-label">Zone</span><span class="popup-value">' + station.zone + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Status</span><span class="popup-value" style="color:' + color + '">' + statusLabel + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Tracks Monitored</span><span class="popup-value">' + station.tracks_monitored + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Active Alerts</span><span class="popup-value">' + alertsText + '</span></div>' +
            '<div class="popup-detail"><span class="popup-label">Daily Trains</span><span class="popup-value">' + station.daily_trains + '</span></div>' +
            '</div>';

        marker.bindPopup(popupHtml, {
            maxWidth: 280,
            className: 'dark-popup',
        });

        // Station name label
        marker.bindTooltip(station.code, {
            permanent: true,
            direction: 'top',
            offset: [0, -12],
            className: 'station-label',
        });
    });

    // ----------------------------------------------------------------
    // Route polylines connecting stations
    // ----------------------------------------------------------------
    routes.forEach(function (route) {
        var color = statusColors[route.status] || '#3b82f6';

        var polyline = L.polyline(route.coordinates, {
            color: color,
            weight: 2.5,
            opacity: 0.6,
            dashArray: '8, 6',
            lineJoin: 'round',
        }).addTo(map);

        // Route tooltip on hover
        polyline.bindTooltip(
            '<strong>' + route.name + '</strong><br>' +
            '<span style="font-size:0.8em;color:#94a3b8">' + route.train + '</span>',
            { sticky: true }
        );
    });

    // ----------------------------------------------------------------
    // Add custom CSS for station labels
    // ----------------------------------------------------------------
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
        '}';
    document.head.appendChild(style);

    // ----------------------------------------------------------------
    // Fit map bounds to show all stations
    // ----------------------------------------------------------------
    if (stations.length > 0) {
        var bounds = L.latLngBounds(
            stations.map(function (s) { return [s.lat, s.lng]; })
        );
        map.fitBounds(bounds, { padding: [40, 40] });
    }
}
