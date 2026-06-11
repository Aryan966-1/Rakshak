// frontend/static/js/train_simulation.js
//
// Rakshak — Train Movement Simulation
// Creates animated train markers that move along railway routes.
// No real GPS — positions are calculated from route geometry
// and time offsets for hackathon demo realism.

'use strict';

/**
 * Initialize train simulation on the Leaflet map.
 * Fetches train positions from the API every 3 seconds
 * and smoothly updates marker positions.
 *
 * @param {L.Map} map - The Leaflet map instance
 */
function initTrainSimulation(map) {
    // Layer group to hold all train markers
    var trainLayer = L.layerGroup().addTo(map);
    var trainMarkers = {};  // Keyed by train ID

    // Train icon using a custom DivIcon
    function createTrainIcon() {
        return L.divIcon({
            html: '<div class="train-marker">🚂</div>',
            className: 'train-icon-wrapper',
            iconSize: [24, 24],
            iconAnchor: [12, 12],
        });
    }

    /**
     * Fetch latest train positions and update markers.
     */
    function updateTrains() {
        fetch('/api/trains/')
            .then(function(response) { return response.json(); })
            .then(function(trains) {
                // Track which trains are still present
                var activeIds = {};

                trains.forEach(function(train) {
                    activeIds[train.id] = true;

                    if (trainMarkers[train.id]) {
                        // Smoothly update existing marker position
                        var currentLatLng = trainMarkers[train.id].getLatLng();
                        var targetLatLng = L.latLng(train.lat, train.lng);

                        // Animate movement over 2.5 seconds
                        _animateMarker(
                            trainMarkers[train.id],
                            currentLatLng,
                            targetLatLng,
                            2500
                        );

                        // Update tooltip
                        trainMarkers[train.id].setTooltipContent(
                            '<strong>' + train.id + '</strong><br>' +
                            '<span style="font-size:0.8em;color:#94a3b8">' +
                            train.speed_kmph + ' km/h</span>'
                        );
                    } else {
                        // Create new marker
                        var marker = L.marker([train.lat, train.lng], {
                            icon: createTrainIcon(),
                            zIndexOffset: 1000,
                        });

                        marker.bindTooltip(
                            '<strong>' + train.id + '</strong><br>' +
                            '<span style="font-size:0.8em;color:#94a3b8">' +
                            train.speed_kmph + ' km/h</span>',
                            {
                                direction: 'top',
                                offset: [0, -14],
                                className: 'station-label',
                            }
                        );

                        marker.bindPopup(
                            '<div class="station-popup">' +
                            '<h4>🚂 ' + train.id + '</h4>' +
                            '<div class="popup-detail"><span class="popup-label">Route</span><span class="popup-value">' + train.route_id + '</span></div>' +
                            '<div class="popup-detail"><span class="popup-label">Speed</span><span class="popup-value">' + train.speed_kmph + ' km/h</span></div>' +
                            '<div class="popup-detail"><span class="popup-label">Progress</span><span class="popup-value">' + Math.round(train.progress * 100) + '%</span></div>' +
                            '</div>',
                            { maxWidth: 250, className: 'dark-popup' }
                        );

                        trainLayer.addLayer(marker);
                        trainMarkers[train.id] = marker;
                    }
                });

                // Remove markers for trains no longer in the response
                Object.keys(trainMarkers).forEach(function(id) {
                    if (!activeIds[id]) {
                        trainLayer.removeLayer(trainMarkers[id]);
                        delete trainMarkers[id];
                    }
                });
            })
            .catch(function(err) {
                console.warn('Train simulation fetch failed:', err);
            });
    }

    /**
     * Smoothly animate a marker from one position to another.
     * Uses requestAnimationFrame for smooth 60fps animation.
     */
    function _animateMarker(marker, from, to, duration) {
        var startTime = performance.now();

        function step(currentTime) {
            var elapsed = currentTime - startTime;
            var progress = Math.min(elapsed / duration, 1.0);

            // Ease-in-out cubic
            var eased = progress < 0.5
                ? 4 * progress * progress * progress
                : 1 - Math.pow(-2 * progress + 2, 3) / 2;

            var lat = from.lat + eased * (to.lat - from.lat);
            var lng = from.lng + eased * (to.lng - from.lng);

            marker.setLatLng([lat, lng]);

            if (progress < 1.0) {
                requestAnimationFrame(step);
            }
        }

        requestAnimationFrame(step);
    }

    // ---- Inject train marker CSS ----
    var style = document.createElement('style');
    style.textContent =
        '.train-icon-wrapper {' +
        '  background: transparent !important;' +
        '  border: none !important;' +
        '}' +
        '.train-marker {' +
        '  font-size: 18px;' +
        '  filter: drop-shadow(0 0 6px rgba(59, 130, 246, 0.6));' +
        '  animation: train-pulse 2s ease-in-out infinite;' +
        '}' +
        '@keyframes train-pulse {' +
        '  0%, 100% { transform: scale(1); }' +
        '  50% { transform: scale(1.15); }' +
        '}';
    document.head.appendChild(style);

    // ---- Start polling ----
    updateTrains();  // Initial fetch
    setInterval(updateTrains, 3000);  // Update every 3 seconds

    console.log('🚂 Train simulation started (3s interval)');
}
