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

// Shared Chart.js defaults — theme-aware
function getChartDefaults() {
    const dark = isDarkMode();
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
                backgroundColor: dark ? '#111111' : '#f0f0f0',
                titleColor: dark ? '#ffffff' : '#000000',
                bodyColor: dark ? '#a0a0a0' : '#555555',
                borderColor: dark ? 'rgba(255,255,255,0.08)' : 'rgba(0,0,0,0.08)',
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
                    color: dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)',
                    drawBorder: false,
                },
                ticks: {
                    color: dark ? '#666666' : '#888888',
                    font: { family: 'JetBrains Mono', size: 10 },
                },
            },
            y: {
                grid: {
                    color: dark ? 'rgba(255,255,255,0.04)' : 'rgba(0,0,0,0.06)',
                    drawBorder: false,
                },
                ticks: {
                    color: dark ? '#666666' : '#888888',
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

// Store chart instances globally to allow destruction
window.rakshakChartInstances = window.rakshakChartInstances || {};

function getChartColorConfig(type, values) {
    if (!values || values.length === 0) return { main: '#10b981', bg: 'rgba(16,185,129,0.2)' };
    const lastValue = values[values.length - 1];
    let status = 'healthy'; // default
    
    if (type === 'vibration') {
        if (lastValue > 5.0) status = 'critical';
        else if (lastValue >= 3.5) status = 'warning';
    } else if (type === 'temperature') {
        if (lastValue > 50) status = 'critical';
        else if (lastValue >= 40) status = 'warning';
    } else if (type === 'gauge') {
        const absVal = Math.abs(lastValue);
        if (absVal > 6) status = 'critical';
        else if (absVal >= 2) status = 'warning';
    }

    if (status === 'critical') return { main: '#ef4444', bg: 'rgba(239,68,68,0.2)' }; // Red
    if (status === 'warning') return { main: '#f59e0b', bg: 'rgba(245,158,11,0.2)' }; // Amber
    return { main: '#10b981', bg: 'rgba(16,185,129,0.2)' }; // Green
}

/**
 * Initialize all three dashboard sensor trend charts.
 * Called from dashboard.html after DOM load with parsed JSON data.
 * @param {Object} data - Sensor trend data with timestamps, vibration, temperature, gauge_deviation
 */
function initDashboardCharts(data) {
    // Only run on pages that have the chart canvases
    if (!document.getElementById('chart-vibration')) return;

    // Store data for re-rendering on theme change
    window.rakshakLastChartData = data;

    // Prevent duplicate chart initialization
    if (window.rakshakChartInstances.vibration) window.rakshakChartInstances.vibration.destroy();
    if (window.rakshakChartInstances.temperature) window.rakshakChartInstances.temperature.destroy();
    if (window.rakshakChartInstances.gauge) window.rakshakChartInstances.gauge.destroy();

    const timestamps = data.timestamps;
    const pointBorder = isDarkMode() ? '#111111' : '#f0f0f0';

    // --- Vibration Chart ---
    const vibCtx = document.getElementById('chart-vibration').getContext('2d');
    const vibColors = getChartColorConfig('vibration', data.vibration);
    window.rakshakChartInstances.vibration = new Chart(vibCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Vibration (mm/s)',
                data: data.vibration,
                borderColor: vibColors.main,
                backgroundColor: createGradient(vibCtx, vibColors.bg, 'rgba(0,0,0,0)'),
                borderWidth: 2,
                pointBackgroundColor: vibColors.main,
                pointBorderColor: pointBorder,
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
        },
    });

    // --- Temperature Chart ---
    const tempCtx = document.getElementById('chart-temperature').getContext('2d');
    const tempColors = getChartColorConfig('temperature', data.temperature);
    window.rakshakChartInstances.temperature = new Chart(tempCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Temperature (°C)',
                data: data.temperature,
                borderColor: tempColors.main,
                backgroundColor: createGradient(tempCtx, tempColors.bg, 'rgba(0,0,0,0)'),
                borderWidth: 2,
                pointBackgroundColor: tempColors.main,
                pointBorderColor: pointBorder,
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
    const gaugeColors = getChartColorConfig('gauge', data.gauge_deviation);
    window.rakshakChartInstances.gauge = new Chart(gaugeCtx, {
        type: 'line',
        data: {
            labels: timestamps,
            datasets: [{
                label: 'Gauge Deviation (mm)',
                data: data.gauge_deviation,
                borderColor: gaugeColors.main,
                backgroundColor: createGradient(gaugeCtx, gaugeColors.bg, 'rgba(0,0,0,0)'),
                borderWidth: 2,
                pointBackgroundColor: gaugeColors.main,
                pointBorderColor: pointBorder,
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
// THEME TOGGLE — Dark/Light mode with localStorage persistence
// ====================================================================
function initThemeToggle() {
    const toggle = document.getElementById('theme-toggle');
    const root = document.documentElement;

    // Determine initial theme: localStorage > system preference > dark (default)
    const stored = localStorage.getItem('rakshak-theme');
    if (stored) {
        root.setAttribute('data-theme', stored);
    } else {
        // Check system preference
        const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
        root.setAttribute('data-theme', prefersDark ? 'dark' : 'light');
    }

    if (!toggle) return;

    toggle.addEventListener('click', function () {
        const current = root.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        root.setAttribute('data-theme', next);
        localStorage.setItem('rakshak-theme', next);

        // Re-render charts if they exist (to update colors)
        if (window.rakshakLastChartData) {
            initDashboardCharts(window.rakshakLastChartData);
        }
    });

    // Listen for system preference changes
    window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', function (e) {
        if (!localStorage.getItem('rakshak-theme')) {
            root.setAttribute('data-theme', e.matches ? 'dark' : 'light');
        }
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
// THEME-AWARE HELPERS
// ====================================================================
function isDarkMode() {
    return document.documentElement.getAttribute('data-theme') !== 'light';
}

// ====================================================================
// INITIALIZE ON DOM READY
// ====================================================================
document.addEventListener('DOMContentLoaded', function () {
    initThemeToggle();
    initLiveClock();
    animateCounters();
    initNavToggle();
});
