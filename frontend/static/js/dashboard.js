// frontend/static/js/dashboard.js
//
// Rakshak — Core JavaScript
// Handles: live clock, KPI counter animations, Chart.js initialization,
//          navigation toggle, and shared utilities.
// This file is loaded on every page via base.html.

'use strict';

// ====================================================================
// LIVE CLOCK — Updates every second with IST time
// ====================================================================
function initLiveClock() {
    const clockTime = document.getElementById('clock-time');
    const clockDate = document.getElementById('clock-date');

    if (!clockTime || !clockDate) return;

    function updateClock() {
        const now = new Date();
        // Format time as HH:MM:SS
        const hours = String(now.getHours()).padStart(2, '0');
        const minutes = String(now.getMinutes()).padStart(2, '0');
        const seconds = String(now.getSeconds()).padStart(2, '0');
        clockTime.textContent = `${hours}:${minutes}:${seconds}`;

        // Format date as YYYY/MM/DD
        const year = now.getFullYear();
        const month = String(now.getMonth() + 1).padStart(2, '0');
        const day = String(now.getDate()).padStart(2, '0');
        clockDate.textContent = `${year}/${month}/${day}`;
    }

    updateClock();
    setInterval(updateClock, 1000);
}

// ====================================================================
// KPI COUNTER ANIMATION — Animates numbers from 0 to target
// ====================================================================
function animateCounters() {
    const counters = document.querySelectorAll('.kpi-value[data-target]');

    counters.forEach(counter => {
        const target = parseFloat(counter.getAttribute('data-target'));
        const duration = 1500; // milliseconds
        const startTime = performance.now();
        const isDecimal = target % 1 !== 0;

        // Format large numbers with Indian locale (Lakhs/Crores)
        function formatValue(val) {
            if (target >= 100000) {
                // Format as Lakhs: 24,50,000 → "24.5L"
                return (val / 100000).toFixed(1) + 'L';
            }
            if (isDecimal) {
                return val.toFixed(1);
            }
            return Math.round(val).toLocaleString('en-IN');
        }

        function step(currentTime) {
            const elapsed = currentTime - startTime;
            const progress = Math.min(elapsed / duration, 1);

            // Ease-out cubic for smooth deceleration
            const eased = 1 - Math.pow(1 - progress, 3);
            const current = eased * target;

            counter.textContent = formatValue(current);

            if (progress < 1) {
                requestAnimationFrame(step);
            }
        }

        requestAnimationFrame(step);
    });
}

// ====================================================================
// CHART.JS INITIALIZATION — Sensor trend charts on the dashboard
// ====================================================================

// Shared Chart.js defaults for dark theme
function getChartDefaults() {
    return {
        responsive: true,
        maintainAspectRatio: false,
        interaction: {
            mode: 'index',
            intersect: false,
        },
        plugins: {
            legend: { display: false },
            tooltip: {
                backgroundColor: '#111827',
                titleColor: '#f1f5f9',
                bodyColor: '#94a3b8',
                borderColor: 'rgba(255,255,255,0.06)',
                borderWidth: 1,
                padding: 12,
                cornerRadius: 8,
                titleFont: { family: 'Inter', weight: '600' },
                bodyFont: { family: 'JetBrains Mono' },
            },
        },
        scales: {
            x: {
                grid: {
                    color: 'rgba(255,255,255,0.04)',
                    drawBorder: false,
                },
                ticks: {
                    color: '#64748b',
                    font: { family: 'JetBrains Mono', size: 10 },
                },
            },
            y: {
                grid: {
                    color: 'rgba(255,255,255,0.04)',
                    drawBorder: false,
                },
                ticks: {
                    color: '#64748b',
                    font: { family: 'JetBrains Mono', size: 10 },
                },
            },
        },
    };
}

/**
 * Create a gradient fill for line charts.
 * @param {CanvasRenderingContext2D} ctx
 * @param {string} colorTop - Top color (RGBA)
 * @param {string} colorBottom - Bottom color (RGBA)
 * @returns {CanvasGradient}
 */
function createGradient(ctx, colorTop, colorBottom) {
    const gradient = ctx.createLinearGradient(0, 0, 0, 200);
    gradient.addColorStop(0, colorTop);
    gradient.addColorStop(1, colorBottom);
    return gradient;
}

/**
 * Initialize all three dashboard sensor trend charts.
 * Called from dashboard.html after DOM load with parsed JSON data.
 * @param {Object} data - Sensor trend data with timestamps, vibration, temperature, gauge_deviation
 */
function initDashboardCharts(data) {
    // Only run on pages that have the chart canvases
    if (!document.getElementById('chart-vibration')) return;

    const timestamps = data.timestamps;

    // --- Vibration Chart ---
    const vibCtx = document.getElementById('chart-vibration').getContext('2d');
    new Chart(vibCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Vibration (mm/s)',
                data: data.vibration,
                borderColor: '#f59e0b',
                backgroundColor: createGradient(vibCtx, 'rgba(245,158,11,0.2)', 'rgba(245,158,11,0)'),
                borderWidth: 2,
                pointBackgroundColor: '#f59e0b',
                pointBorderColor: '#111827',
                pointBorderWidth: 2,
                pointRadius: 3,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4,
            }],
        },
        options: {
            ...getChartDefaults(),
            scales: {
                ...getChartDefaults().scales,
                y: {
                    ...getChartDefaults().scales.y,
                    min: 0,
                    max: 7,
                    ticks: {
                        ...getChartDefaults().scales.y.ticks,
                        stepSize: 1,
                    },
                },
            },
            plugins: {
                ...getChartDefaults().plugins,
                // Warning threshold annotation via plugin
                annotation: {
                    annotations: {
                        warningLine: {
                            type: 'line',
                            yMin: 3.5,
                            yMax: 3.5,
                            borderColor: 'rgba(245, 158, 11, 0.4)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                        },
                        criticalLine: {
                            type: 'line',
                            yMin: 5.0,
                            yMax: 5.0,
                            borderColor: 'rgba(239, 68, 68, 0.4)',
                            borderWidth: 1,
                            borderDash: [5, 5],
                        },
                    },
                },
            },
        },
    });

    // --- Temperature Chart ---
    const tempCtx = document.getElementById('chart-temperature').getContext('2d');
    new Chart(tempCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Temperature (°C)',
                data: data.temperature,
                borderColor: '#ef4444',
                backgroundColor: createGradient(tempCtx, 'rgba(239,68,68,0.2)', 'rgba(239,68,68,0)'),
                borderWidth: 2,
                pointBackgroundColor: '#ef4444',
                pointBorderColor: '#111827',
                pointBorderWidth: 2,
                pointRadius: 3,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4,
            }],
        },
        options: {
            ...getChartDefaults(),
            scales: {
                ...getChartDefaults().scales,
                y: {
                    ...getChartDefaults().scales.y,
                    min: 20,
                    max: 60,
                    ticks: {
                        ...getChartDefaults().scales.y.ticks,
                        stepSize: 10,
                    },
                },
            },
        },
    });

    // --- Gauge Deviation Chart ---
    const gaugeCtx = document.getElementById('chart-gauge').getContext('2d');
    new Chart(gaugeCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Gauge Deviation (mm)',
                data: data.gauge_deviation,
                borderColor: '#3b82f6',
                backgroundColor: createGradient(gaugeCtx, 'rgba(59,130,246,0.2)', 'rgba(59,130,246,0)'),
                borderWidth: 2,
                pointBackgroundColor: '#3b82f6',
                pointBorderColor: '#111827',
                pointBorderWidth: 2,
                pointRadius: 3,
                pointHoverRadius: 6,
                fill: true,
                tension: 0.4,
            }],
        },
        options: {
            ...getChartDefaults(),
            scales: {
                ...getChartDefaults().scales,
                y: {
                    ...getChartDefaults().scales.y,
                    min: 0,
                    max: 4,
                    ticks: {
                        ...getChartDefaults().scales.y.ticks,
                        stepSize: 0.5,
                    },
                },
            },
        },
    });
}

// ====================================================================
// NAVIGATION TOGGLE (mobile)
// ====================================================================
function initNavToggle() {
    const toggle = document.getElementById('nav-toggle');
    const navInner = document.querySelector('.nav-inner');

    if (!toggle || !navInner) return;

    toggle.addEventListener('click', function () {
        navInner.classList.toggle('nav-inner--open');
        toggle.classList.toggle('nav-toggle--active');
    });
}

// ====================================================================
// INITIALIZE ON DOM READY
// ====================================================================
document.addEventListener('DOMContentLoaded', function () {
    initLiveClock();
    animateCounters();
    initNavToggle();
});
