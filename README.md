# README.md
# 🛡️ RAKSHAK — AI-Powered Predictive Rail Maintenance System

> **FAR AWAY 2026** | Predicting failures before derailments happen.

---

## Problem Statement

Indian Railways operates one of the world's largest rail networks — 68,000+ route kilometers carrying over 23 million passengers daily. Despite this scale, infrastructure monitoring remains largely **manual and reactive**:

- Track inspections rely on periodic visual checks by gangers
- Rail fractures, gauge deviations, and thermal buckling are detected **after** damage occurs
- Emergency speed restrictions and derailments cause massive economic losses (₹30,000+ Crore/year)

**There is no unified, real-time predictive system** that detects failures before they become catastrophic.

## Solution

**Rakshak** is an AI-powered predictive maintenance platform that:

1. **Monitors** track infrastructure via IoT sensor telemetry (vibration, temperature, gauge)
2. **Detects** anomalies in real-time using ML-based anomaly detection
3. **Predicts** failures 72 hours before they occur
4. **Dispatches** maintenance crews automatically via intelligent ticketing
5. **Visualizes** the entire railway network on an interactive operations dashboard

## Current Status

### ✅ Phase 1 — UI Prototype (Current)

A fully functional, demo-ready **Railway Operations Control Center** dashboard built with Django.

**Features implemented:**
- Dashboard with 5 KPI cards (Health, Alerts, Failures, Savings, Tracks)
- Sensor trend charts (vibration, temperature, gauge deviation)
- Alerts page with severity filtering
- Maintenance tickets with assigned engineer details
- Interactive railway map (Leaflet.js) with 12 Indian stations
- Dark operations-center theme with responsive design
- All mock data simulates realistic Indian Railways operations

**Not yet implemented (future phases):**
- Real sensor ingestion
- ML anomaly detection (Isolation Forest)
- AI agent layer (LangGraph/CrewAI)
- PostgreSQL + TimescaleDB
- Authentication & RBAC
- Docker containerization

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | HTML, CSS, Vanilla JavaScript |
| Charts | Chart.js |
| Maps | Leaflet.js |
| Backend | Django |
| Database | SQLite (prototype) → PostgreSQL (future) |
| Fonts | Inter, JetBrains Mono (Google Fonts) |

## Project Structure

```
Rakshak/
├── frontend/
│   ├── templates/
│   │   ├── base.html              # Base layout with nav + header
│   │   ├── dashboard.html         # Dashboard with KPIs + charts
│   │   ├── alerts.html            # Alert listing + filters
│   │   ├── tickets.html           # Maintenance ticket management
│   │   └── map.html               # Leaflet railway map
│   └── static/
│       ├── css/
│       │   └── dashboard.css      # Complete dark theme CSS
│       ├── js/
│       │   ├── dashboard.js       # Charts, clock, counters
│       │   └── map.js             # Leaflet map initialization
│       └── images/
│
├── backend/
│   ├── manage.py
│   ├── rakshak_project/
│   │   ├── settings.py            # Django configuration
│   │   ├── urls.py                # Root URL routing
│   │   ├── wsgi.py
│   │   └── asgi.py
│   ├── core/                      # Shared utilities
│   │   └── context_processors.py  # Navigation + project meta
│   ├── sensors/                   # Dashboard + sensor data
│   │   ├── views.py
│   │   ├── mock_data.py
│   │   └── urls.py
│   ├── alerts/                    # Alert management
│   │   ├── views.py
│   │   ├── mock_data.py
│   │   └── urls.py
│   ├── tickets/                   # Maintenance tickets
│   │   ├── views.py
│   │   ├── mock_data.py
│   │   └── urls.py
│   ├── map_view/                  # Railway map view
│   │   ├── views.py
│   │   ├── mock_data.py
│   │   └── urls.py
│   └── agents/                    # Placeholder — future AI agents
│       └── __init__.py
│
├── docs/
│   ├── architecture/
│   │   └── system_overview.md
│   └── reports/
│       └── PHASE_REPORT.md
│
├── presentation/
├── demo_assets/
│   └── demo_scenario.md
│
├── .gitignore
├── README.md
└── requirements.txt
```

## Quick Start

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/Rakshak.git
cd Rakshak

# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (macOS/Linux)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Run the Server

```bash
cd backend
python manage.py migrate
python manage.py runserver
```

### Access the Dashboard

Open your browser and navigate to:

| Page | URL |
|------|-----|
| Dashboard | http://127.0.0.1:8000/ |
| Alerts | http://127.0.0.1:8000/alerts/ |
| Tickets | http://127.0.0.1:8000/tickets/ |
| Map | http://127.0.0.1:8000/map/ |

## Team

**FAR AWAY 2026**

## License

This project is developed for the FAR AWAY 2026 hackathon.