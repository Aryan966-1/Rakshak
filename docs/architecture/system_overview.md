<!-- docs/architecture/system_overview.md -->
# Rakshak — System Architecture Overview

## Phase 1 Architecture (Current)

Phase 1 is a **monolithic Django application** serving server-rendered HTML pages with hardcoded mock data. No database models, no API layer, no ML pipelines.

```
┌─────────────────────────────────────────────────────────┐
│                      BROWSER                            │
│  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐     │
│  │Dashboard│ │ Alerts  │ │Tickets  │ │   Map   │     │
│  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘     │
│       │           │           │           │            │
│  Chart.js    Vanilla JS   Vanilla JS   Leaflet.js     │
└───────┼───────────┼───────────┼───────────┼────────────┘
        │           │           │           │
        ▼           ▼           ▼           ▼
┌─────────────────────────────────────────────────────────┐
│                   DJANGO SERVER                         │
│                                                         │
│  ┌──────────────────────────────────────────────────┐  │
│  │              URL Router (urls.py)                 │  │
│  │  /           → sensors.views.dashboard            │  │
│  │  /alerts/    → alerts.views.alerts_page           │  │
│  │  /tickets/   → tickets.views.tickets_page         │  │
│  │  /map/       → map_view.views.map_page            │  │
│  └──────────────────────────────────────────────────┘  │
│                         │                               │
│  ┌──────────────────────┼──────────────────────────┐   │
│  │              Views Layer                         │   │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐        │   │
│  │  │ sensors  │ │  alerts  │ │ tickets  │ ...    │   │
│  │  │ views.py │ │ views.py │ │ views.py │        │   │
│  │  └────┬─────┘ └────┬─────┘ └────┬─────┘        │   │
│  │       │             │             │               │   │
│  │  ┌────▼─────┐ ┌────▼─────┐ ┌────▼─────┐        │   │
│  │  │mock_data │ │mock_data │ │mock_data │        │   │
│  │  │   .py    │ │   .py    │ │   .py    │        │   │
│  │  └──────────┘ └──────────┘ └──────────┘        │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Core Context Processors                  │   │
│  │  navigation() — Nav items for all pages          │   │
│  │  project_meta() — Branding, version, timestamp   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Template Engine (Django Templates)       │   │
│  │  base.html → dashboard.html / alerts.html / ... │   │
│  └─────────────────────────────────────────────────┘   │
│                                                         │
│  ┌─────────────────────────────────────────────────┐   │
│  │         Static Files                             │   │
│  │  css/dashboard.css  js/dashboard.js  js/map.js  │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Data Flow

1. User navigates to a URL (e.g., `/alerts/`)
2. Django URL router dispatches to the appropriate view
3. View imports hardcoded mock data from `mock_data.py`
4. View passes data as template context
5. Django template renders HTML with the data
6. JavaScript (Chart.js, Leaflet) initializes with data from `json_script` tags
7. CSS provides the dark operations-center theme

## Future Architecture (Phase 2+)

```
┌──────────────────────────────────────────────────────────────┐
│                        BROWSER                                │
│   Dashboard │ Alerts │ Tickets │ Map │ Analytics              │
└──────────────────────┬───────────────────────────────────────┘
                       │ REST API / WebSocket
┌──────────────────────▼───────────────────────────────────────┐
│                    API GATEWAY                                │
│              Django + Django REST Framework                    │
│              JWT Authentication + RBAC                         │
├──────────────────────────────────────────────────────────────┤
│                   SERVICE LAYER                               │
│  ┌──────────┐  ┌───────────┐  ┌──────────┐  ┌───────────┐  │
│  │ Sensors  │  │  Alerts   │  │ Tickets  │  │  Agents   │  │
│  │ Service  │  │  Service  │  │ Service  │  │(LangGraph)│  │
│  └────┬─────┘  └─────┬─────┘  └────┬─────┘  └─────┬─────┘  │
│       │              │              │              │          │
├───────┼──────────────┼──────────────┼──────────────┼──────────┤
│       ▼              ▼              ▼              ▼          │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │              DATA LAYER                                  │ │
│  │  PostgreSQL + TimescaleDB │ Redis Cache │ ML Model Store │ │
│  └─────────────────────────────────────────────────────────┘ │
├──────────────────────────────────────────────────────────────┤
│                   ML PIPELINE                                 │
│  Isolation Forest │ LSTM │ Anomaly Detection │ Prediction    │
├──────────────────────────────────────────────────────────────┤
│                  INFRASTRUCTURE                               │
│           Docker │ Docker Compose │ CI/CD                     │
└──────────────────────────────────────────────────────────────┘
```

## Design Decisions (Phase 1)

| Decision | Rationale |
|----------|-----------|
| No DRF in Phase 1 | Standard Django views sufficient for server-rendered pages |
| Mock data in Python dicts | Fastest path to a working prototype; easy to swap for DB later |
| CDN for Chart.js & Leaflet | No build step needed; reduces project complexity |
| Single CSS file | Manageable at current scale; component extraction in Phase 2 |
| No authentication | Not needed for demo; will add Django auth + JWT in Phase 2 |
| SQLite default | Zero-config; no Postgres setup for reviewers running locally |
| Context processors for nav | DRY pattern — nav/branding injected into all templates automatically |
