<!-- docs/reports/PHASE_REPORT.md -->
# Rakshak — Phase 1 Report

**Date:** 2026-06-08
**Phase:** 1 — UI Prototype
**Status:** ✅ Complete

---

## Files Created

| # | File | Purpose |
|---|------|---------|
| 1 | `requirements.txt` | Python dependencies (Django 4.2 only) |
| 2 | `.gitignore` | Git exclusions for Python/Django/IDE artifacts |
| 3 | `README.md` | Project overview, problem statement, setup instructions |
| 4 | `backend/manage.py` | Django management CLI entry point |
| 5 | `backend/rakshak_project/__init__.py` | Django project package |
| 6 | `backend/rakshak_project/settings.py` | Django config (templates, static, apps, timezone) |
| 7 | `backend/rakshak_project/urls.py` | Root URL routing to all 4 apps |
| 8 | `backend/rakshak_project/wsgi.py` | WSGI application entry point |
| 9 | `backend/rakshak_project/asgi.py` | ASGI application entry point |
| 10 | `backend/core/__init__.py` | Core shared utilities package |
| 11 | `backend/core/context_processors.py` | Navigation + project metadata injected into all templates |
| 12 | `backend/sensors/__init__.py` | Sensors app package |
| 13 | `backend/sensors/views.py` | Dashboard view — passes KPI data and sensor trends to template |
| 14 | `backend/sensors/mock_data.py` | Hardcoded sensor data: KPIs, 24h trends, track sections, readings |
| 15 | `backend/sensors/urls.py` | Dashboard URL route at `/` |
| 16 | `backend/alerts/__init__.py` | Alerts app package |
| 17 | `backend/alerts/views.py` | Alerts view with severity filtering via query string |
| 18 | `backend/alerts/mock_data.py` | 12 realistic alerts across severity levels and railway zones |
| 19 | `backend/alerts/urls.py` | Alerts URL route at `/alerts/` |
| 20 | `backend/tickets/__init__.py` | Tickets app package |
| 21 | `backend/tickets/views.py` | Tickets view with status filtering |
| 22 | `backend/tickets/mock_data.py` | 10 maintenance tickets with Indian engineer names and P.Way details |
| 23 | `backend/tickets/urls.py` | Tickets URL route at `/tickets/` |
| 24 | `backend/map_view/__init__.py` | Map view app package |
| 25 | `backend/map_view/views.py` | Map view — passes station/route data as JSON for Leaflet |
| 26 | `backend/map_view/mock_data.py` | 12 Indian stations with GPS coords + 4 rail routes |
| 27 | `backend/map_view/urls.py` | Map URL route at `/map/` |
| 28 | `backend/agents/__init__.py` | Placeholder — agent layer reserved for future phases |
| 29 | `frontend/templates/base.html` | Base layout: header, nav, footer, CDN scripts |
| 30 | `frontend/templates/dashboard.html` | Dashboard: 5 KPI cards, 3 charts, track table, readings |
| 31 | `frontend/templates/alerts.html` | Alert list with severity badges and filter controls |
| 32 | `frontend/templates/tickets.html` | Ticket table (desktop) + card layout (mobile) |
| 33 | `frontend/templates/map.html` | Full-height Leaflet map with legend and stats bar |
| 34 | `frontend/static/css/dashboard.css` | Complete dark theme CSS (700+ lines, custom design system) |
| 35 | `frontend/static/js/dashboard.js` | Live clock, KPI animations, Chart.js init, nav toggle |
| 36 | `frontend/static/js/map.js` | Leaflet map: dark tiles, markers, routes, popups |
| 37 | `frontend/static/images/.gitkeep` | Preserve empty images directory |
| 38 | `docs/architecture/system_overview.md` | Architecture diagrams and design decisions |
| 39 | `docs/reports/PHASE_REPORT.md` | This file |
| 40 | `demo_assets/demo_scenario.md` | 5-minute demo script for judge presentations |
| 41 | `presentation/.gitkeep` | Preserve presentation directory |

**Total: 41 files created**

---

## Architecture Decisions

| Decision | Rationale |
|----------|-----------|
| **No Django REST Framework** | Standard Django views sufficient for server-rendered prototype. APIs deferred to Phase 2. |
| **Mock data as Python dicts** | Fastest path to working prototype. Easy to swap for ORM queries when DB is added. |
| **CDN for Chart.js & Leaflet** | Eliminates build tooling. Chart.js 4.4.4 and Leaflet 1.9.4 loaded from stable CDNs. |
| **Single monolithic CSS** | Manageable at 700 lines. Component extraction planned for Phase 2 when page count grows. |
| **Context processors for nav** | DRY pattern — nav items and branding injected into every template via Django middleware. |
| **SQLite (default)** | Zero-config for developers and judges. PostgreSQL migration planned for Phase 2. |
| **No authentication** | Not needed for demo. Django auth + JWT planned for Phase 2. |
| **`json_script` for JS data** | Django's built-in safe serialization — avoids XSS while passing data to Chart.js/Leaflet. |
| **Separate `map_view` app** | The spec's `agents/` directory is reserved for future AI agents; map logic lives in its own app. |
| **Indian locale data** | Realistic station names, coordinates, engineer names, railway zones, and P.Way terminology build credibility. |

---

## Verification Results

| Check | Result |
|-------|--------|
| `python manage.py check` | ✅ 0 issues |
| `python manage.py migrate` | ✅ Success |
| Dashboard (HTTP 200) | ✅ |
| Alerts page (HTTP 200) | ✅ |
| Tickets page (HTTP 200) | ✅ |
| Map page (HTTP 200) | ✅ |
| All KPI cards rendered | ✅ |
| Chart canvases present | ✅ |
| CSS loaded | ✅ |
| JS loaded | ✅ |
| Mock data serialized | ✅ |
| Leaflet map container | ✅ |
| Station/route JSON injected | ✅ |

---

## Known Limitations

1. **No persistent data** — All data is hardcoded. Refreshing the page always shows the same state.
2. **No real-time updates** — The live clock updates, but sensor data does not change dynamically.
3. **No authentication** — No login, no role-based access control.
4. **No API layer** — No DRF endpoints for mobile apps or third-party integrations.
5. **No database models** — Cannot create, update, or delete alerts/tickets.
6. **CDN dependency** — Chart.js and Leaflet require internet to load. No offline fallback.
7. **No automated tests** — No Django test suite yet.
8. **Single-user** — No concept of user sessions or multi-tenancy.

---

## Technical Debt

- CSS could be split into component files as the project grows
- Mock data should be centralized or moved to fixtures when DB is introduced
- No error handling for template rendering edge cases
- No logging or monitoring
- Static files not production-optimized (no minification, no cache busting)

---

## Next Phase Roadmap

### Phase 2 — Backend Foundation
- [ ] PostgreSQL + Django models for sensors, alerts, tickets
- [ ] Django REST Framework API endpoints
- [ ] Django authentication (login, RBAC)
- [ ] Real sensor data ingestion (simulated IoT pipeline)
- [ ] WebSocket for live dashboard updates
- [ ] Automated test suite

### Phase 3 — ML & Intelligence
- [ ] Isolation Forest anomaly detection model
- [ ] LSTM time-series prediction for failure forecasting
- [ ] Model training pipeline
- [ ] Prediction confidence scores in dashboard

### Phase 4 — Agent Layer
- [ ] LangGraph/CrewAI multi-agent orchestration
- [ ] Automated ticket creation from anomaly detection
- [ ] Self-healing workflow agents
- [ ] Natural language query interface

### Phase 5 — Production
- [ ] Docker + Docker Compose
- [ ] TimescaleDB for time-series data
- [ ] Redis caching layer
- [ ] CI/CD pipeline
- [ ] Production deployment (cloud)
