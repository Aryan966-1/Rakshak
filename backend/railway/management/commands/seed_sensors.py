import random
from datetime import timedelta
from decimal import Decimal

from django.core.management.base import BaseCommand
from django.db import transaction
from django.utils import timezone

from railway.models import (
    Asset,
    Sensor,
    SensorReading,
    SensorType,
    TrackSection,
)


class Command(BaseCommand):
    help = "Seed Sensor types, sensors, and timeseries sensor readings."

    def add_arguments(self, parser):
        parser.add_argument(
            "--reset",
            action="store_true",
            help="Delete existing Sensor data before seeding.",
        )

    @transaction.atomic
    def handle(self, *args, **options):
        if options["reset"]:
            self.stdout.write(self.style.WARNING("Resetting sensor data…"))
            SensorReading.objects.all().delete()
            Sensor.objects.all().delete()
            SensorType.objects.all().delete()

        tracks = list(TrackSection.objects.all())
        if not tracks:
            self.stderr.write(
                self.style.ERROR(
                    "No TrackSections found. Run 'python manage.py seed_routes' first."
                )
            )
            return

        random.seed(42)  # For reproducible random data

        # 1. Create Sensor Types
        sensor_types_data = [
            {
                "name": "Vibration",
                "measurement_description": "Measures track vibration and shock",
                "measurement_unit": "mm/s",
                "normal_min": Decimal("0.0"),
                "normal_max": Decimal("3.0"),
                "critical_min": None,
                "critical_max": Decimal("5.0"),
            },
            {
                "name": "Temperature",
                "measurement_description": "Measures rail temperature",
                "measurement_unit": "°C",
                "normal_min": Decimal("5.0"),
                "normal_max": Decimal("40.0"),
                "critical_min": Decimal("0.0"),
                "critical_max": Decimal("50.0"),
            },
            {
                "name": "Gauge Deviation",
                "measurement_description": "Measures deviation from standard broad gauge",
                "measurement_unit": "mm",
                "normal_min": Decimal("-1.0"),
                "normal_max": Decimal("1.0"),
                "critical_min": Decimal("-2.5"),
                "critical_max": Decimal("2.5"),
            },
        ]

        sensor_types = {}
        for st_data in sensor_types_data:
            st, _ = SensorType.objects.get_or_create(
                name=st_data["name"],
                defaults=st_data,
            )
            sensor_types[st.name] = st

        self.stdout.write(f"Created/Verified {len(sensor_types)} SensorTypes.")

        # 2. Create Sensors for TrackSections
        sensors_created = 0
        all_sensors = []
        for track in tracks:
            # Create a default Track asset for the section if one doesn't exist
            asset, _ = Asset.objects.get_or_create(
                asset_code=f"AST-{track.section_code}",
                defaults={
                    "track_section": track,
                    "asset_type": "track",
                    "latitude": track.start_station.latitude,
                    "longitude": track.start_station.longitude,
                }
            )

            # Attach 1-3 sensors to each track via the asset
            types_to_attach = random.sample(
                list(sensor_types.values()),
                random.randint(1, 3)
            )
            for st in types_to_attach:
                # Generate a unique serial number
                prefix = {"Vibration": "VIB", "Temperature": "TMP", "Gauge Deviation": "GAU"}.get(st.name, "SNR")
                serial = f"{track.section_code}-{prefix}-{random.randint(100, 999)}"
                
                sensor, created = Sensor.objects.get_or_create(
                    serial_number=serial,
                    defaults={
                        "sensor_type": st,
                        "asset": asset,
                        "sensor_code": serial,
                        "health_status": "healthy" if random.random() > 0.1 else "degraded",
                        "installation_date": timezone.now().date() - timedelta(days=random.randint(30, 365)),
                    }
                )
                if created:
                    sensors_created += 1
                all_sensors.append(sensor)

        self.stdout.write(f"Created {sensors_created} Sensors.")

        # 3. Create Sensor Readings (24h time-series)
        now = timezone.now()
        readings_to_create = []
        # 1 reading every 4 hours for the last 24 hours -> 6 readings per sensor
        timestamps = [now - timedelta(hours=h) for h in range(0, 24, 4)]
        
        for sensor in all_sensors:
            st = sensor.sensor_type
            # Decide the health status of this sensor to make trends
            # 80% healthy, 15% warning, 5% critical
            health_state = random.random()
            
            for ts in timestamps:
                if st.name == "Vibration":
                    if health_state < 0.8:
                        val = random.uniform(0.5, 2.5) # Healthy
                    elif health_state < 0.95:
                        val = random.uniform(2.5, 4.5) # Warning
                    else:
                        val = random.uniform(4.5, 6.5) # Critical
                elif st.name == "Temperature":
                    if health_state < 0.8:
                        val = random.uniform(20.0, 35.0)
                    elif health_state < 0.95:
                        val = random.uniform(35.0, 48.0)
                    else:
                        val = random.uniform(48.0, 60.0)
                else: # Gauge Deviation
                    if health_state < 0.8:
                        val = random.uniform(-0.5, 0.5)
                    elif health_state < 0.95:
                        val = random.choice([random.uniform(1.0, 2.0), random.uniform(-2.0, -1.0)])
                    else:
                        val = random.choice([random.uniform(2.5, 4.0), random.uniform(-4.0, -2.5)])
                
                readings_to_create.append(
                    SensorReading(
                        sensor=sensor,
                        recorded_at=ts,
                        raw_value=Decimal(str(round(val, 2))),
                        anomaly_flag=False # Can be improved later
                    )
                )

        if readings_to_create:
            SensorReading.objects.bulk_create(readings_to_create, ignore_conflicts=True)
            self.stdout.write(f"Bulk created {len(readings_to_create)} SensorReadings.")

        self.stdout.write(self.style.SUCCESS("[OK] Sensor seeding complete."))
