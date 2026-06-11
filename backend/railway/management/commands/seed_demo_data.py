"""
railway/management/commands/seed_demo_data.py
Seed command — generates 1000 tickets, 50 alerts, and maintenance teams.

Usage:
    python manage.py seed_demo_data          # insert
    python manage.py seed_demo_data --reset   # wipe & re-insert

All records are linked to valid stations, track sections, zones,
and divisions. No orphan records are created.
"""

import random
from datetime import timedelta
from decimal import Decimal

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from railway.models import (
    Alert,
    Division,
    MaintenanceTeam,
    Station,
    Ticket,
    TrackSection,
    Zone,
)


# ===================================================================
# ALERT TEMPLATES — realistic Indian Railway infrastructure alerts
# ===================================================================
ALERT_TEMPLATES = [
    {"type": "anomaly",          "sev": "critical", "title": "Rail Fracture Detected",
     "desc": "Ultrasonic sensor detected micro-fracture on rail head. Immediate speed restriction recommended."},
    {"type": "threshold_breach", "sev": "critical", "title": "Excessive Rail Temperature",
     "desc": "Rail temperature exceeding 52°C. Risk of rail buckling. Speed restriction applied."},
    {"type": "anomaly",          "sev": "critical", "title": "Ballast Washout Risk",
     "desc": "Heavy rainfall detected. Ballast erosion sensors triggered on embankment section."},
    {"type": "threshold_breach", "sev": "critical", "title": "Signal Cable Damage",
     "desc": "Track-circuit signal cable damaged. Manual caution order issued."},
    {"type": "prediction",       "sev": "critical", "title": "Bearing Failure Predicted",
     "desc": "ML model predicts axle bearing failure within 48 hours. Inspection required."},
    {"type": "anomaly",          "sev": "warning",  "title": "Gauge Deviation Warning",
     "desc": "Rail gauge deviation of +3.2mm detected. Track geometry degrading."},
    {"type": "anomaly",          "sev": "warning",  "title": "Vibration Anomaly",
     "desc": "Elevated vibration levels detected. Possible ballast settlement or loose fastener."},
    {"type": "threshold_breach", "sev": "warning",  "title": "Turnout Wear Detected",
     "desc": "Switch rail wear exceeding threshold. Replacement scheduling advised."},
    {"type": "prediction",       "sev": "warning",  "title": "Weld Joint Stress Elevated",
     "desc": "Thermit weld joint showing stress concentration. Strain gauge reading 15% above baseline."},
    {"type": "anomaly",          "sev": "warning",  "title": "Fishplate Bolt Loosening",
     "desc": "IoT torque sensor detected loosening fishplate bolts."},
    {"type": "prediction",       "sev": "warning",  "title": "OHE Wire Sag Predicted",
     "desc": "Overhead wire sag predicted due to thermal expansion. Monitor closely."},
    {"type": "system",           "sev": "info",     "title": "Scheduled Inspection Due",
     "desc": "Routine ultrasonic flaw detection due. Last inspection: 48 hours ago."},
    {"type": "system",           "sev": "info",     "title": "Sensor Calibration Complete",
     "desc": "All vibration sensors recalibrated successfully. Readings verified."},
    {"type": "manual",           "sev": "info",     "title": "Track Realignment Complete",
     "desc": "Track realignment completed by Gang. Post-tamping readings nominal."},
    {"type": "system",           "sev": "info",     "title": "Weather Advisory Active",
     "desc": "Heavy monsoon rain forecast. Track patrol frequency increased."},
]


# ===================================================================
# TICKET TEMPLATES
# ===================================================================
TICKET_TITLES = [
    "Rail Fracture — Emergency Weld Repair",
    "Track Geometry Correction",
    "Ballast Replenishment",
    "Fishplate Replacement",
    "Signal Cable Repair",
    "OHE Wire Tension Adjustment",
    "Rail Grinding (Surface Defect)",
    "Turnout Switch Replacement",
    "Bridge Pier Inspection",
    "Level Crossing Gate Repair",
    "Sleeper Replacement (Damaged)",
    "Drainage Clearing — Monsoon Prep",
    "Rail Temperature Monitoring Setup",
    "Ultrasonic Flaw Detection Run",
    "Vegetation Clearing — Visibility",
    "Speed Restriction Enforcement",
    "Track Circuit Battery Replacement",
    "Axle Counter Calibration",
    "Point Machine Lubrication",
    "Ballast Cleaning (Machine)",
]

# Indian engineer names for team leads
TEAM_LEAD_NAMES = [
    "Rajesh Kumar", "Suresh Sharma", "Vikram Singh", "Arun Patel",
    "Deepak Verma", "Manoj Tiwari", "Sanjay Gupta", "Amit Mishra",
    "Rahul Yadav", "Pradeep Joshi", "Satish Reddy", "Kiran Nair",
    "Vinod Chandra", "Ravi Shankar", "Ganesh Iyer", "Sunil Desai",
    "Naveen Babu", "Ashok Pillai", "Mohan Rao", "Prakash Hegde",
]

TEAM_SPECIALIZATIONS = ["Track", "Signal", "Electrical", "Bridge", "General"]


class Command(BaseCommand):
    help = "Generate demo data: maintenance teams, 50 alerts, 1000 tickets."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete existing demo data before seeding.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        if options["reset"]:
            self.stdout.write(self.style.WARNING("Resetting demo data…"))
            Ticket.objects.all().delete()
            Alert.objects.all().delete()
            MaintenanceTeam.objects.all().delete()

        random.seed(42)

        # --- Load prerequisite data ---
        track_sections = list(TrackSection.objects.select_related(
            "start_station__division__zone",
            "end_station__division__zone",
        ).all())

        if not track_sections:
            self.stderr.write(self.style.ERROR(
                "No track sections found. Run seed_routes first."
            ))
            return

        divisions = list(Division.objects.all())
        if not divisions:
            self.stderr.write(self.style.ERROR(
                "No divisions found. Run seed_master_data first."
            ))
            return

        # ---- Maintenance Teams ----
        self.stdout.write("Creating maintenance teams…")
        teams = []
        team_created = 0
        for i, div in enumerate(divisions):
            for spec in random.sample(TEAM_SPECIALIZATIONS, k=min(2, len(TEAM_SPECIALIZATIONS))):
                team_code = f"MT-{div.code}-{spec[:3].upper()}-{i+1:03d}"
                lead = random.choice(TEAM_LEAD_NAMES)
                obj, created = MaintenanceTeam.objects.get_or_create(
                    team_code=team_code,
                    defaults={
                        "team_name": f"{div.name} {spec} Team",
                        "division": div,
                        "specialization": spec,
                        "team_lead_name": lead,
                        "contact_phone": f"+91-{random.randint(70000,99999)}{random.randint(10000,99999)}",
                    },
                )
                teams.append(obj)
                if created:
                    team_created += 1
        self.stdout.write(f"  Teams: {team_created} created")

        # ---- Alerts (50) ----
        self.stdout.write("Creating 50 alerts…")
        now = timezone.now()
        alerts = []
        alert_created = 0

        alert_statuses = ["active"] * 35 + ["acknowledged"] * 8 + ["resolved"] * 7

        for i in range(50):
            template = random.choice(ALERT_TEMPLATES)
            ts = random.choice(track_sections)
            alert_code = f"ALT-{now.year}-{i+1:04d}"

            generated_at = now - timedelta(
                hours=random.randint(0, 168),
                minutes=random.randint(0, 59),
            )

            status = alert_statuses[i % len(alert_statuses)]

            obj, created = Alert.objects.get_or_create(
                alert_code=alert_code,
                defaults={
                    "track_section": ts,
                    "alert_type": template["type"],
                    "severity": template["sev"],
                    "title": template["title"],
                    "description": (
                        f"{template['desc']} "
                        f"Section: {ts.start_station.station_name} — "
                        f"{ts.end_station.station_name}."
                    ),
                    "confidence_score": (
                        Decimal(str(round(random.uniform(0.7, 0.99), 4)))
                        if template["type"] != "manual" else None
                    ),
                    "generated_at": generated_at,
                    "status": status,
                    "acknowledged_at": (
                        generated_at + timedelta(minutes=random.randint(5, 60))
                        if status in ("acknowledged", "resolved") else None
                    ),
                    "resolved_at": (
                        generated_at + timedelta(hours=random.randint(1, 24))
                        if status == "resolved" else None
                    ),
                    "generated_by": (
                        "ml_model" if template["type"] == "prediction"
                        else "sensor" if template["type"] in ("anomaly", "threshold_breach")
                        else "manual" if template["type"] == "manual"
                        else "system"
                    ),
                },
            )
            alerts.append(obj)
            if created:
                alert_created += 1

        self.stdout.write(f"  Alerts: {alert_created} created")

        # ---- Tickets (1000) ----
        self.stdout.write("Creating 1000 tickets…")
        ticket_created = 0

        priorities = ["critical"] * 100 + ["high"] * 250 + ["medium"] * 400 + ["low"] * 250
        statuses = (
            ["open"] * 200 + ["assigned"] * 250 + ["in_progress"] * 200 +
            ["scheduled"] * 150 + ["resolved"] * 150 + ["closed"] * 50
        )

        for i in range(1000):
            ts = random.choice(track_sections)
            team = random.choice(teams)
            alert = random.choice(alerts) if random.random() < 0.3 else None
            title = random.choice(TICKET_TITLES)
            priority = priorities[i % len(priorities)]
            status = statuses[i % len(statuses)]

            ticket_code = f"TKT-{i+1:04d}"
            created_at_offset = timedelta(
                hours=random.randint(0, 720),
                minutes=random.randint(0, 59),
            )

            est_hours = Decimal(str(round(random.uniform(1, 48), 2)))
            cost_est = Decimal(str(random.randint(5000, 500000)))

            _, created = Ticket.objects.get_or_create(
                ticket_code=ticket_code,
                defaults={
                    "alert": alert,
                    "track_section": ts,
                    "assigned_team": team if status != "open" else None,
                    "title": f"{title} — {ts.start_station.station_name}",
                    "description": (
                        f"{title} required on section "
                        f"{ts.start_station.station_name} — "
                        f"{ts.end_station.station_name} "
                        f"({ts.start_station.division.zone.name})."
                    ),
                    "priority": priority,
                    "status": status,
                    "scheduled_for": (
                        now + timedelta(hours=random.randint(1, 72))
                        if status == "scheduled" else None
                    ),
                    "estimated_duration_hours": est_hours,
                    "resolved_at": (
                        now - timedelta(hours=random.randint(1, 48))
                        if status in ("resolved", "closed") else None
                    ),
                    "resolution_notes": (
                        "Work completed. Section cleared for traffic."
                        if status in ("resolved", "closed") else ""
                    ),
                    "cost_estimate_inr": cost_est,
                    "cost_actual_inr": (
                        cost_est * Decimal(str(round(random.uniform(0.8, 1.3), 2)))
                        if status in ("resolved", "closed") else None
                    ),
                },
            )
            if created:
                ticket_created += 1

        self.stdout.write(f"  Tickets: {ticket_created} created")

        self.stdout.write(self.style.SUCCESS(
            f"\n[OK] Demo data seeded: {team_created} teams, "
            f"{alert_created} alerts, {ticket_created} tickets."
        ))
