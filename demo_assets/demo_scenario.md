<!-- demo_assets/demo_scenario.md -->
# Rakshak — Demo Scenario Script

## Demo Setup

1. Start the Django server: `cd backend && python manage.py runserver`
2. Open browser to `http://127.0.0.1:8000/`
3. Ensure browser window is maximized for best visual impact

---

## Demo Flow (5–7 minutes)

### Scene 1: The Problem (30 seconds)

**Narration:**
> "Every year, Indian Railways faces over 50 major derailments. Most are caused by track defects that could have been detected days in advance — rail fractures, thermal buckling, gauge deviations. The problem? Detection is manual, periodic, and reactive."

### Scene 2: Dashboard Overview (90 seconds)

**Navigate to:** `http://127.0.0.1:8000/`

**Show:**
- Point to the live timestamp — "This is a real-time operations center."
- Walk through the 5 KPI cards:
  - **87.3% Overall Health** — "Our system monitors 156 track sections across 8 railway zones."
  - **12 Active Alerts** — "Right now, 12 infrastructure issues have been detected."
  - **3 Predicted Failures** — "Our AI predicts 3 potential failures in the next 72 hours."
  - **₹24.5L Cost Savings** — "Predictive maintenance saves ₹24.5 Lakhs vs reactive repairs."
  - **156 Tracks Monitored** — "Coverage spans Delhi to Chennai, Mumbai to Kolkata."

**Show sensor trend charts:**
- "These charts show real-time sensor telemetry — vibration, temperature, and gauge deviation."
- "Notice the vibration spike at 14:00 — that's what triggered the critical alert on the Howrah section."

**Show track health table:**
- "Each track section has a computed health score. TRK-HWH-004 (Howrah–Bandel) is at 45% — critical."

### Scene 3: Alerts (60 seconds)

**Navigate to:** `http://127.0.0.1:8000/alerts/`

**Show:**
- "The alerts page gives operations staff a prioritized view of all infrastructure issues."
- Click the **Critical** filter — "4 critical alerts right now."
- Highlight `ALT-2026-001`: "Rail fracture detected on the Howrah section by ultrasonic sensors."
- Highlight `ALT-2026-007`: "Ballast washout risk due to heavy rainfall — our sensors detected embankment erosion."

### Scene 4: Maintenance Tickets (60 seconds)

**Navigate to:** `http://127.0.0.1:8000/tickets/`

**Show:**
- "Every alert automatically generates a maintenance ticket assigned to the right engineer."
- Point to TKT-2026-001: "Rajesh Kumar, Senior Section Engineer, is already en route with Gang 7."
- Show the ETA column: "Each ticket has estimated repair time — this one is 4 hours."
- "The system tracks Open → In Progress → Resolved lifecycle."

### Scene 5: Railway Map (60 seconds)

**Navigate to:** `http://127.0.0.1:8000/map/`

**Show:**
- "The map view gives a bird's-eye view of our entire monitored network."
- Point to color-coded markers: "Green = healthy, yellow = warning, red = critical."
- Click on Howrah station marker — show the popup with 4 active alerts.
- "The dashed lines show monitored rail routes — notice the Delhi–Howrah route is red."

### Scene 6: The Vision (30 seconds)

**Narration:**
> "What you've seen is Phase 1 — the prototype. In Phase 2, we connect real IoT sensors. In Phase 3, we deploy Isolation Forest anomaly detection and multi-agent AI for automated response. Rakshak doesn't just monitor railways — it predicts failures before they happen, saving lives and crores."

---

## Key Talking Points

- **Predictive, not reactive** — detect failures 72 hours before they occur
- **Indian Railways scale** — designed for 68,000+ km of track
- **Cost savings** — predictive maintenance costs 1/5th of emergency repairs
- **Real engineer workflow** — tickets auto-assigned to P.Way engineers by zone
- **Open architecture** — Django backend ready for ML pipeline integration
