"""
railway/models.py
Rakshak — Railway Predictive Maintenance & Monitoring Platform

Production-grade schema: 18 concrete models + 1 abstract base.
Single source of truth for sensor ingestion, alert generation,
ticket management, ML predictions, and GIS visualization.

Layers:
    1. Core Geography   — Zone, Division, Station
    2. Infrastructure   — TrackSection, Asset
    3. Sensor           — SensorType, Sensor, SensorCalibration, SensorReading
    4. Alert            — Alert, AlertEscalation
    5. Maintenance      — MaintenanceTeam, Ticket, TicketStatusLog
    6. ML               — MLModel, MLModelRun, AnomalyPrediction
    7. Audit            — AuditLog

Constraints:
    - PROTECT on all critical infrastructure FK targets
    - DecimalField for all measurements and coordinates
    - UUIDField for sensor hardware identity
    - SQLite compatible, PostgreSQL ready
"""

import uuid
from decimal import Decimal

from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone


# ===================================================================
# ABSTRACT BASE
# ===================================================================

class TimeStampedModel(models.Model):
    """
    Abstract base providing created_at / updated_at timestamps.

    Every concrete model except AuditLog inherits this.
    """

    created_at = models.DateTimeField(
        auto_now_add=True,
        help_text="Row creation timestamp (set once on INSERT).",
    )
    updated_at = models.DateTimeField(
        auto_now=True,
        help_text="Row last-modification timestamp (updated on every save).",
    )

    class Meta:
        abstract = True


# ===================================================================
# CORE GEOGRAPHY
# ===================================================================

class Zone(TimeStampedModel):
    """
    Indian Railways top-level organizational unit.

    There are 18 railway zones in India (e.g., Northern Railway, Central
    Railway). Every Division, and by extension every Station, belongs to
    exactly one Zone.
    """

    code = models.CharField(
        max_length=10,
        unique=True,
        help_text="Official zone code, e.g. 'NR', 'CR', 'SR'.",
    )
    name = models.CharField(
        max_length=100,
        help_text="Full zone name, e.g. 'Northern Railway'.",
    )
    headquarters = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="City where the zone HQ is located.",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="False if the zone has been reorganized or dissolved.",
    )

    class Meta:
        db_table = "rakshak_zone"
        ordering = ["name"]
        verbose_name = "Zone"
        verbose_name_plural = "Zones"
        indexes = [
            models.Index(fields=["is_active"], name="idx_zone_active"),
        ]

    def __str__(self):
        return f"{self.code} — {self.name}"


class Division(TimeStampedModel):
    """
    Mid-level administrative unit within a Zone.

    Hierarchy: Zone → Division → Station.
    Each division manages stations, track sections, and maintenance
    teams within its jurisdiction.
    """

    zone = models.ForeignKey(
        Zone,
        on_delete=models.PROTECT,
        related_name="divisions",
        help_text="Parent zone. PROTECT prevents deleting a zone that has divisions.",
    )
    code = models.CharField(
        max_length=10,
        unique=True,
        help_text="Division code, e.g. 'DLI', 'MUM'.",
    )
    name = models.CharField(
        max_length=100,
        help_text="Full division name, e.g. 'Delhi Division'.",
    )
    headquarters = models.CharField(
        max_length=100,
        blank=True,
        default="",
    )
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "rakshak_division"
        ordering = ["zone", "name"]
        verbose_name = "Division"
        verbose_name_plural = "Divisions"
        indexes = [
            models.Index(fields=["is_active"], name="idx_div_active"),
        ]

    def __str__(self):
        return f"{self.code} — {self.name}"


class Station(TimeStampedModel):
    """
    Physical railway station within a Division.

    Stores GPS coordinates for GIS visualization and operational
    metadata (junction/terminal flags, daily traffic).
    """

    station_code = models.CharField(
        max_length=10,
        unique=True,
        help_text="Official station code, e.g. 'NDLS', 'CSMT'.",
    )
    station_name = models.CharField(
        max_length=100,
        help_text="Full station name, e.g. 'New Delhi'.",
    )
    division = models.ForeignKey(
        Division,
        on_delete=models.PROTECT,
        related_name="stations",
        help_text="Parent division. PROTECT prevents deleting a division with stations.",
    )
    latitude = models.DecimalField(
        max_digits=9,
        decimal_places=6,
        validators=[MinValueValidator(Decimal("-90")), MaxValueValidator(Decimal("90"))],
        help_text="GPS latitude. DecimalField(9,6) covers ±90.000000.",
    )
    longitude = models.DecimalField(
        max_digits=9,
        decimal_places=6,
        validators=[MinValueValidator(Decimal("-180")), MaxValueValidator(Decimal("180"))],
        help_text="GPS longitude. DecimalField(9,6) covers ±180.000000.",
    )
    elevation_m = models.DecimalField(
        max_digits=7,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Elevation above sea level in metres.",
    )
    is_junction = models.BooleanField(
        default=False,
        help_text="True if the station has converging/diverging routes.",
    )
    is_terminal = models.BooleanField(
        default=False,
        help_text="True if the station is a terminus (end-of-line).",
    )
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "rakshak_station"
        ordering = ["station_name"]
        verbose_name = "Station"
        verbose_name_plural = "Stations"
        indexes = [
            models.Index(fields=["is_active"], name="idx_sta_active"),
            models.Index(fields=["is_junction"], name="idx_sta_junction"),
        ]

    def __str__(self):
        return f"{self.station_code} — {self.station_name}"


# ===================================================================
# INFRASTRUCTURE
# ===================================================================

class TrackSection(TimeStampedModel):
    """
    A segment of railway track between two stations.

    Core infrastructure entity. Assets (bridges, signals, OHE) are
    deployed on track sections; sensors are deployed on those assets.
    Health is NOT stored — it is derived from readings, alerts, and
    ML predictions at query time.
    """

    class Direction(models.TextChoices):
        UP = "up", "Up Direction"
        DOWN = "down", "Down Direction"
        BOTH = "both", "Bidirectional"

    class TrackType(models.TextChoices):
        BROAD_GAUGE = "broad_gauge", "Broad Gauge (1676 mm)"
        METRE_GAUGE = "metre_gauge", "Metre Gauge (1000 mm)"
        NARROW_GAUGE = "narrow_gauge", "Narrow Gauge (762 mm)"

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        UNDER_MAINTENANCE = "under_maintenance", "Under Maintenance"
        CLOSED = "closed", "Closed"
        DECOMMISSIONED = "decommissioned", "Decommissioned"

    section_code = models.CharField(
        max_length=20,
        unique=True,
        help_text="Unique track section code, e.g. 'TRK-NDL-001'.",
    )
    start_station = models.ForeignKey(
        Station,
        on_delete=models.PROTECT,
        related_name="track_sections_starting",
        help_text="Station at the start of this section.",
    )
    end_station = models.ForeignKey(
        Station,
        on_delete=models.PROTECT,
        related_name="track_sections_ending",
        help_text="Station at the end of this section.",
    )
    direction = models.CharField(
        max_length=10,
        choices=Direction.choices,
        default=Direction.BOTH,
    )
    track_type = models.CharField(
        max_length=15,
        choices=TrackType.choices,
        default=TrackType.BROAD_GAUGE,
        help_text="Rail gauge type. Indian Railways is predominantly broad gauge.",
    )
    length_km = models.DecimalField(
        max_digits=7,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Section length in kilometres.",
    )
    max_speed_kmph = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Maximum permissible speed in km/h on this section.",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.ACTIVE,
    )
    commissioned_date = models.DateField(
        null=True,
        blank=True,
        help_text="Date the section was first commissioned.",
    )
    last_major_inspection = models.DateField(
        null=True,
        blank=True,
    )
    next_scheduled_inspection = models.DateField(
        null=True,
        blank=True,
        help_text="Used by maintenance scheduling queries.",
    )

    class Meta:
        db_table = "rakshak_track_section"
        ordering = ["section_code"]
        verbose_name = "Track Section"
        verbose_name_plural = "Track Sections"
        constraints = [
            models.UniqueConstraint(
                fields=[
                    "start_station",
                    "end_station",
                    "direction"
                ],
                name="uniq_track_route_direction"
            ),
        ]
        indexes = [
            models.Index(fields=["status"], name="idx_ts_status"),
            models.Index(
                fields=["next_scheduled_inspection"],
                name="idx_ts_next_insp",
            ),
        ]

    def __str__(self):
        return self.section_code


class Asset(TimeStampedModel):
    """
    A physical asset deployed on a track section.

    Provides the intermediate layer between TrackSection and Sensor:
      TrackSection → Asset → Sensor

    This enables precision localization: a sensor is on a specific
    bridge, signal, OHE structure, or rail segment — not just
    "somewhere on the track."
    """

    class AssetType(models.TextChoices):
        TRACK = "track", "Track"
        BRIDGE = "bridge", "Bridge"
        SIGNAL = "signal", "Signal"
        OHE = "ohe", "OHE"
        CROSSING = "crossing", "Crossing"
        OTHER = "other", "Other"

    class Status(models.TextChoices):
        OPERATIONAL = "operational", "Operational"
        UNDER_REPAIR = "under_repair", "Under Repair"
        DECOMMISSIONED = "decommissioned", "Decommissioned"

    asset_code = models.CharField(
        max_length=30,
        unique=True,
        help_text="Unique asset identifier, e.g. 'AST-BRG-NDL-001'.",
    )
    track_section = models.ForeignKey(
        TrackSection,
        on_delete=models.PROTECT,
        related_name="assets",
        help_text="Parent track section. PROTECT prevents deleting a section with assets.",
    )
    asset_type = models.CharField(
        max_length=10,
        choices=AssetType.choices,
        help_text="Category of infrastructure asset.",
    )
    latitude = models.DecimalField(
        max_digits=9,
        decimal_places=6,
        null=True,
        blank=True,
        validators=[MinValueValidator(Decimal("-90")), MaxValueValidator(Decimal("90"))],
        help_text="GPS latitude of the asset.",
    )
    longitude = models.DecimalField(
        max_digits=9,
        decimal_places=6,
        null=True,
        blank=True,
        validators=[MinValueValidator(Decimal("-180")), MaxValueValidator(Decimal("180"))],
        help_text="GPS longitude of the asset.",
    )
    status = models.CharField(
        max_length=20,
        choices=Status.choices,
        default=Status.OPERATIONAL,
    )
    description = models.TextField(
        blank=True,
        default="",
        help_text="Free-text description (e.g., bridge span, signal type).",
    )

    class Meta:
        db_table = "rakshak_asset"
        ordering = ["track_section", "asset_type", "asset_code"]
        verbose_name = "Asset"
        verbose_name_plural = "Assets"
        indexes = [
            models.Index(
                fields=["asset_type", "status"],
                name="idx_asset_type_status",
            ),
        ]

    def __str__(self):
        return f"{self.asset_code} ({self.get_asset_type_display()})"


# ===================================================================
# SENSOR LAYER
# ===================================================================

class SensorType(TimeStampedModel):
    """
    Lookup table defining sensor categories and their measurement
    characteristics.

    Provides threshold metadata consumed by alert-generation logic
    and ML feature engineering.
    """

    name = models.CharField(
        max_length=50,
        unique=True,
        help_text="Sensor type name, e.g. 'Vibration', 'Temperature'.",
    )
    measurement_unit = models.CharField(
        max_length=20,
        help_text="SI unit of measurement, e.g. 'mm/s', '°C', 'mm'.",
    )
    measurement_description = models.TextField(
        blank=True,
        default="",
        help_text="Human-readable explanation of what is measured.",
    )
    normal_min = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Lower bound of the normal operating range.",
    )
    normal_max = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Upper bound of the normal operating range.",
    )
    critical_min = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Below this value, the reading is critical.",
    )
    critical_max = models.DecimalField(
        max_digits=12,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Above this value, the reading is critical.",
    )
    default_sampling_rate_hz = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Default sampling frequency in Hz for this sensor type.",
    )

    class Meta:
        db_table = "rakshak_sensor_type"
        ordering = ["name"]
        verbose_name = "Sensor Type"
        verbose_name_plural = "Sensor Types"

    def __str__(self):
        return self.name


class Sensor(TimeStampedModel):
    """
    Individual sensor device deployed on an Asset.

    Hierarchy: TrackSection → Asset → Sensor → SensorReading.

    Each sensor has a globally unique UUID (sensor_uid) for hardware
    identification independent of the auto-incrementing PK.
    """

    class HealthStatus(models.TextChoices):
        HEALTHY = "healthy", "Healthy"
        DEGRADED = "degraded", "Degraded"
        FAULTY = "faulty", "Faulty"
        OFFLINE = "offline", "Offline"

    sensor_uid = models.UUIDField(
        default=uuid.uuid4,
        unique=True,
        editable=False,
        help_text="Globally unique hardware identifier (UUID v4). Immutable after creation.",
    )
    sensor_code = models.CharField(
        max_length=30,
        unique=True,
        help_text="Human-readable sensor code, e.g. 'SEN-VIB-NDL-001'.",
    )
    sensor_type = models.ForeignKey(
        SensorType,
        on_delete=models.PROTECT,
        related_name="sensors",
        help_text="Measurement type this sensor produces. PROTECT prevents orphaning sensors.",
    )
    asset = models.ForeignKey(
        Asset,
        on_delete=models.PROTECT,
        related_name="sensors",
        help_text="Physical asset this sensor is mounted on. PROTECT prevents orphaning sensors.",
    )
    manufacturer = models.CharField(
        max_length=100,
        blank=True,
        default="",
    )
    model_number = models.CharField(
        max_length=50,
        blank=True,
        default="",
    )
    serial_number = models.CharField(
        max_length=100,
        blank=True,
        default="",
    )
    firmware_version = models.CharField(
        max_length=30,
        blank=True,
        default="",
    )
    installation_date = models.DateField(
        null=True,
        blank=True,
    )
    sampling_rate_hz = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Actual sampling rate, may differ from sensor_type default.",
    )
    health_status = models.CharField(
        max_length=10,
        choices=HealthStatus.choices,
        default=HealthStatus.HEALTHY,
    )
    last_heartbeat = models.DateTimeField(
        null=True,
        blank=True,
        help_text="Last time the sensor reported it was alive.",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="False if the sensor is decommissioned or removed.",
    )

    class Meta:
        db_table = "rakshak_sensor"
        ordering = ["sensor_code"]
        verbose_name = "Sensor"
        verbose_name_plural = "Sensors"
        indexes = [
            models.Index(fields=["is_active"], name="idx_sen_active"),
            models.Index(fields=["health_status"], name="idx_sen_health"),
            models.Index(
                fields=["asset", "sensor_type"],
                name="idx_sen_asset_type",
            ),
        ]

    def __str__(self):
        return self.sensor_code


class SensorCalibration(TimeStampedModel):
    """
    Calibration event record for a sensor.

    Append-mostly table: each row represents one calibration event.
    The most recent calibration defines the active offset/scale.
    """

    sensor = models.ForeignKey(
        Sensor,
        on_delete=models.PROTECT,
        related_name="calibrations",
        help_text="Calibrated sensor. PROTECT prevents deleting sensors with calibration history.",
    )
    calibrated_at = models.DateTimeField(
        help_text="When the calibration was performed.",
    )
    calibrated_by = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Technician name or identifier. CharField(150) matches auth.User.username.",
    )
    offset_applied = models.DecimalField(
        max_digits=12,
        decimal_places=6,
        default=Decimal("0"),
        help_text="Additive offset applied to raw sensor readings.",
    )
    scale_factor = models.DecimalField(
        max_digits=12,
        decimal_places=6,
        default=Decimal("1"),
        help_text="Multiplicative scale factor applied to raw sensor readings.",
    )
    notes = models.TextField(
        blank=True,
        default="",
    )
    next_calibration_due = models.DateField(
        null=True,
        blank=True,
        help_text="Scheduled date for the next calibration.",
    )

    class Meta:
        db_table = "rakshak_sensor_calibration"
        ordering = ["-calibrated_at"]
        verbose_name = "Sensor Calibration"
        verbose_name_plural = "Sensor Calibrations"
        indexes = [
            models.Index(
                fields=["next_calibration_due"],
                name="idx_cal_next_due",
            ),
            models.Index(
                fields=["sensor", "calibrated_at"],
                name="idx_cal_sensor_time",
            ),
        ]

    def __str__(self):
        return f"Cal {self.sensor_id} @ {self.calibrated_at:%Y-%m-%d}"


class SensorReading(TimeStampedModel):
    """
    Individual sensor measurement — the highest-volume table.

    THIS IS THE MOST IMPORTANT TABLE IN THE SYSTEM.

    Each row represents one reading from one sensor at one point in
    time. The schema supports raw + processed values, quality scoring,
    anomaly flagging (set by ML pipeline), and extensible JSON metrics.

    Hierarchy: TrackSection → Asset → Sensor → SensorReading.

    UniqueConstraint on (sensor, recorded_at) prevents duplicate
    readings from sensor re-transmission or ingestion retries.
    """

    sensor = models.ForeignKey(
        Sensor,
        on_delete=models.PROTECT,
        related_name="readings",
        help_text="Source sensor. PROTECT prevents deleting sensors with historical readings.",
    )
    recorded_at = models.DateTimeField(
        db_index=True,
        help_text=(
            "Timestamp when the sensor physically took the measurement. "
            "Distinct from created_at which records ingestion time."
        ),
    )

    # --- Measurement values ---
    raw_value = models.DecimalField(
        max_digits=14,
        decimal_places=4,
        help_text="Unprocessed sensor output. DecimalField(14,4) covers ±10 billion.",
    )
    processed_value = models.DecimalField(
        max_digits=14,
        decimal_places=4,
        null=True,
        blank=True,
        help_text="Post-calibration / filtered value. Null if no processing applied.",
    )

    # --- Quality and anomaly ---
    quality_score = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        null=True,
        blank=True,
        validators=[MinValueValidator(Decimal("0")), MaxValueValidator(Decimal("1"))],
        help_text="Data quality metric, 0.0000 (garbage) to 1.0000 (perfect).",
    )
    anomaly_flag = models.BooleanField(
        default=False,
        help_text="Set to True by the ML pipeline when an anomaly is detected.",
    )
    anomaly_score = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        null=True,
        blank=True,
        validators=[MinValueValidator(Decimal("0")), MaxValueValidator(Decimal("1"))],
        help_text="ML-assigned anomaly confidence, 0.0000 (normal) to 1.0000 (certain anomaly).",
    )

    # --- Network / ingestion metadata ---
    transmission_latency_ms = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Time between sensor recording and system receipt, in milliseconds.",
    )

    # --- Extensible metrics ---
    extra_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=(
            "Extensible key-value bag for additional per-reading metrics. "
            "Example: {'rms': 5.2, 'peak_frequency_hz': 120.5}."
        ),
    )

    class Meta:
        db_table = "rakshak_sensor_reading"
        # No default ordering — high-volume table; let queries specify.
        verbose_name = "Sensor Reading"
        verbose_name_plural = "Sensor Readings"
        constraints = [
            models.UniqueConstraint(
                fields=["sensor", "recorded_at"],
                name="uniq_reading_sensor_time",
            ),
        ]
        indexes = [
            models.Index(
                fields=["sensor", "recorded_at"],
                name="idx_reading_sensor_time",
            ),
            models.Index(
                fields=["recorded_at"],
                name="idx_reading_time",
            ),
            models.Index(
                fields=["anomaly_flag"],
                name="idx_reading_anomaly",
            ),
        ]

    def __str__(self):
        return f"Reading {self.pk} @ {self.recorded_at}"


# ===================================================================
# ALERT LAYER
# ===================================================================

class Alert(TimeStampedModel):
    """
    Infrastructure alert raised by sensors, ML models, or operators.

    An alert is always anchored to a TrackSection. It may optionally
    be linked to the specific Sensor and SensorReading that triggered
    it (nullable for manual / system alerts).
    """

    class AlertType(models.TextChoices):
        ANOMALY = "anomaly", "Anomaly Detected"
        THRESHOLD_BREACH = "threshold_breach", "Threshold Breach"
        PREDICTION = "prediction", "Predictive Alert"
        MANUAL = "manual", "Manually Raised"
        SYSTEM = "system", "System Alert"

    class Severity(models.TextChoices):
        CRITICAL = "critical", "Critical"
        WARNING = "warning", "Warning"
        INFO = "info", "Info"

    class Status(models.TextChoices):
        ACTIVE = "active", "Active"
        ACKNOWLEDGED = "acknowledged", "Acknowledged"
        RESOLVED = "resolved", "Resolved"
        DISMISSED = "dismissed", "Dismissed"

    class GeneratedBy(models.TextChoices):
        SYSTEM = "system", "System"
        ML_MODEL = "ml_model", "ML Model"
        MANUAL = "manual", "Manual"
        SENSOR = "sensor", "Sensor"

    alert_code = models.CharField(
        max_length=30,
        unique=True,
        help_text="Unique alert identifier, e.g. 'ALT-2026-001'.",
    )
    trigger_reading = models.ForeignKey(
        SensorReading,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="triggered_alerts",
        help_text="The specific sensor reading that triggered this alert, if any.",
    )
    sensor = models.ForeignKey(
        "Sensor",
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="alerts",
        help_text="The sensor that produced the triggering data, if applicable.",
    )
    track_section = models.ForeignKey(
        TrackSection,
        on_delete=models.PROTECT,
        related_name="alerts",
        help_text="Track section where the alert condition exists. PROTECT.",
    )
    alert_type = models.CharField(
        max_length=20,
        choices=AlertType.choices,
    )
    severity = models.CharField(
        max_length=10,
        choices=Severity.choices,
    )
    title = models.CharField(max_length=200)
    description = models.TextField(
        blank=True,
        default="",
    )
    confidence_score = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        null=True,
        blank=True,
        validators=[MinValueValidator(Decimal("0")), MaxValueValidator(Decimal("1"))],
        help_text="Confidence in the alert, 0.0000–1.0000. Null for manual alerts.",
    )
    generated_at = models.DateTimeField(
        help_text="When the alert condition was detected.",
    )
    status = models.CharField(
        max_length=15,
        choices=Status.choices,
        default=Status.ACTIVE,
    )
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    generated_by = models.CharField(
        max_length=10,
        choices=GeneratedBy.choices,
        default=GeneratedBy.SYSTEM,
        help_text="What generated this alert: system logic, ML model, manual, or sensor firmware.",
    )

    class Meta:
        db_table = "rakshak_alert"
        ordering = ["-generated_at"]
        verbose_name = "Alert"
        verbose_name_plural = "Alerts"
        indexes = [
            models.Index(
                fields=["status", "severity"],
                name="idx_alert_status_sev",
            ),
            models.Index(
                fields=["generated_at"],
                name="idx_alert_generated",
            ),
            models.Index(
                fields=["track_section", "status"],
                name="idx_alert_ts_status",
            ),
        ]

    def __str__(self):
        return f"{self.alert_code} — {self.title}"


class AlertEscalation(TimeStampedModel):
    """
    Records a severity escalation event on an Alert.

    Tracks who escalated, from/to severity, and why.
    CASCADE: if the parent alert is deleted, escalation logs go with it.
    """

    alert = models.ForeignKey(
        Alert,
        on_delete=models.CASCADE,
        related_name="escalations",
        help_text="Parent alert. CASCADE: escalations are deleted with their alert.",
    )
    escalated_at = models.DateTimeField(
        help_text="When the escalation occurred.",
    )
    escalated_by = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Person or system that performed the escalation.",
    )
    escalated_to = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Role or person the alert was escalated to.",
    )
    from_severity = models.CharField(
        max_length=10,
        choices=Alert.Severity.choices,
    )
    to_severity = models.CharField(
        max_length=10,
        choices=Alert.Severity.choices,
    )
    reason = models.TextField(
        blank=True,
        default="",
    )

    class Meta:
        db_table = "rakshak_alert_escalation"
        ordering = ["-escalated_at"]
        verbose_name = "Alert Escalation"
        verbose_name_plural = "Alert Escalations"
        indexes = [
            models.Index(fields=["escalated_at"], name="idx_esc_time"),
        ]

    def __str__(self):
        return (
            f"Esc {self.alert_id}: "
            f"{self.from_severity} → {self.to_severity}"
        )


# ===================================================================
# MAINTENANCE LAYER
# ===================================================================

class MaintenanceTeam(TimeStampedModel):
    """
    A field maintenance team assigned to a Division.

    Teams are assigned to Tickets for on-site inspection and repair.
    """

    team_code = models.CharField(
        max_length=20,
        unique=True,
        help_text="Unique team identifier, e.g. 'MT-DLI-001'.",
    )
    team_name = models.CharField(
        max_length=100,
        help_text="Human-readable team name.",
    )
    division = models.ForeignKey(
        Division,
        on_delete=models.PROTECT,
        related_name="maintenance_teams",
        help_text="Division this team belongs to. PROTECT.",
    )
    specialization = models.CharField(
        max_length=50,
        blank=True,
        default="",
        help_text="Team specialization, e.g. 'Track', 'Signal', 'Electrical'.",
    )
    team_lead_name = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Name of the team lead. CharField(150) for future FK→User migration.",
    )
    contact_phone = models.CharField(
        max_length=20,
        blank=True,
        default="",
    )
    is_active = models.BooleanField(default=True)

    class Meta:
        db_table = "rakshak_maintenance_team"
        ordering = ["team_name"]
        verbose_name = "Maintenance Team"
        verbose_name_plural = "Maintenance Teams"
        indexes = [
            models.Index(fields=["is_active"], name="idx_team_active"),
        ]

    def __str__(self):
        return f"{self.team_code} — {self.team_name}"


class Ticket(TimeStampedModel):
    """
    Maintenance work ticket, optionally linked to an Alert.

    Tracks the full lifecycle from creation through assignment,
    scheduling, and resolution — including cost tracking in INR.
    """

    class Priority(models.TextChoices):
        CRITICAL = "critical", "Critical"
        HIGH = "high", "High"
        MEDIUM = "medium", "Medium"
        LOW = "low", "Low"

    class Status(models.TextChoices):
        OPEN = "open", "Open"
        ASSIGNED = "assigned", "Assigned"
        IN_PROGRESS = "in_progress", "In Progress"
        SCHEDULED = "scheduled", "Scheduled"
        RESOLVED = "resolved", "Resolved"
        CLOSED = "closed", "Closed"

    ticket_code = models.CharField(
        max_length=20,
        unique=True,
        help_text="Unique ticket identifier, e.g. 'TKT-001'.",
    )
    alert = models.ForeignKey(
        Alert,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="tickets",
        help_text="Originating alert. Nullable for manually created tickets.",
    )
    track_section = models.ForeignKey(
        TrackSection,
        on_delete=models.PROTECT,
        related_name="tickets",
        help_text="Track section where work is required. PROTECT.",
    )
    assigned_team = models.ForeignKey(
        MaintenanceTeam,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="tickets",
        help_text="Team assigned to this ticket. Nullable until assignment.",
    )
    title = models.CharField(max_length=200)
    description = models.TextField(
        blank=True,
        default="",
    )
    priority = models.CharField(
        max_length=10,
        choices=Priority.choices,
        default=Priority.MEDIUM,
    )
    status = models.CharField(
        max_length=15,
        choices=Status.choices,
        default=Status.OPEN,
    )
    scheduled_for = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the maintenance work is scheduled to begin.",
    )
    estimated_duration_hours = models.DecimalField(
        max_digits=6,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Estimated duration of maintenance work in hours.",
    )
    resolved_at = models.DateTimeField(null=True, blank=True)
    resolution_notes = models.TextField(
        blank=True,
        default="",
    )
    cost_estimate_inr = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Estimated cost in INR.",
    )
    cost_actual_inr = models.DecimalField(
        max_digits=14,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Actual cost incurred in INR.",
    )

    class Meta:
        db_table = "rakshak_ticket"
        ordering = ["-created_at"]
        verbose_name = "Ticket"
        verbose_name_plural = "Tickets"
        indexes = [
            models.Index(
                fields=["status", "priority"],
                name="idx_tkt_status_pri",
            ),
            models.Index(
                fields=["scheduled_for"],
                name="idx_tkt_scheduled",
            ),
            models.Index(
                fields=["track_section", "status"],
                name="idx_tkt_ts_status",
            ),
        ]

    def __str__(self):
        return f"{self.ticket_code} — {self.title}"


class TicketStatusLog(TimeStampedModel):
    """
    Append-only status transition log for a Ticket.

    Records every state change with who made it and why.
    CASCADE: if the parent ticket is deleted, its logs go with it.
    """

    ticket = models.ForeignKey(
        Ticket,
        on_delete=models.CASCADE,
        related_name="status_logs",
        help_text="Parent ticket. CASCADE: logs are deleted with their ticket.",
    )
    from_status = models.CharField(
        max_length=15,
        choices=Ticket.Status.choices,
    )
    to_status = models.CharField(
        max_length=15,
        choices=Ticket.Status.choices,
    )
    changed_by = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Person or system that changed the status.",
    )
    changed_at = models.DateTimeField(
        help_text="When the status change occurred.",
    )
    notes = models.TextField(
        blank=True,
        default="",
    )

    class Meta:
        db_table = "rakshak_ticket_status_log"
        ordering = ["-changed_at"]
        verbose_name = "Ticket Status Log"
        verbose_name_plural = "Ticket Status Logs"
        indexes = [
            models.Index(fields=["changed_at"], name="idx_tsl_changed"),
        ]

    def __str__(self):
        return f"{self.ticket_id}: {self.from_status} → {self.to_status}"


# ===================================================================
# ML LAYER
# ===================================================================

class MLModel(TimeStampedModel):
    """
    Registry of machine learning models deployed in the Rakshak
    pipeline.

    Provides stable FK targets for the ML team. A model can have
    multiple versions; only one version per model should be active
    at a time (enforced by application logic, not DB constraint).
    """

    model_name = models.CharField(
        max_length=100,
        help_text="Model name, e.g. 'IsolationForest', 'LSTM-Vibration'.",
    )
    model_version = models.CharField(
        max_length=30,
        help_text="Semantic version string, e.g. '1.0.0', '2.3.1-beta'.",
    )
    model_type = models.CharField(
        max_length=50,
        help_text="Algorithm family, e.g. 'isolation_forest', 'lstm', 'xgboost'.",
    )
    target_sensor_types = models.JSONField(
        default=list,
        blank=True,
        help_text=(
            "List of SensorType names this model targets. "
            "Example: ['Vibration', 'Temperature']."
        ),
    )
    performance_metrics = models.JSONField(
        default=dict,
        blank=True,
        help_text=(
            "Evaluation metrics snapshot. "
            "Example: {'precision': 0.92, 'recall': 0.88, 'f1': 0.90}."
        ),
    )
    hyperparameters = models.JSONField(
        default=dict,
        blank=True,
        help_text="Model hyperparameters at training time.",
    )
    is_active = models.BooleanField(
        default=True,
        help_text="Whether this model version is currently deployed.",
    )

    class Meta:
        db_table = "rakshak_ml_model"
        ordering = ["model_name", "model_version"]
        verbose_name = "ML Model"
        verbose_name_plural = "ML Models"
        constraints = [
            models.UniqueConstraint(
                fields=["model_name", "model_version"],
                name="uniq_model_name_version",
            ),
        ]
        indexes = [
            models.Index(fields=["is_active"], name="idx_ml_active"),
        ]

    def __str__(self):
        return f"{self.model_name} v{self.model_version}"


class MLModelRun(TimeStampedModel):
    """
    A single execution of an ML model against a dataset.

    Tracks run lifecycle (pending → running → completed/failed),
    processing stats, and optional association with a track section.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        COMPLETED = "completed", "Completed"
        FAILED = "failed", "Failed"

    model = models.ForeignKey(
        MLModel,
        on_delete=models.PROTECT,
        related_name="runs",
        help_text="ML model that was executed. PROTECT preserves run history.",
    )
    track_section = models.ForeignKey(
        TrackSection,
        on_delete=models.SET_NULL,
        null=True,
        blank=True,
        related_name="ml_runs",
        help_text="Track section analyzed, if the run was scoped to one section.",
    )
    run_started_at = models.DateTimeField(
        help_text="When the model run began.",
    )
    run_completed_at = models.DateTimeField(
        null=True,
        blank=True,
        help_text="When the model run finished (null if still running or failed).",
    )
    readings_processed = models.PositiveIntegerField(
        default=0,
        help_text="Number of sensor readings fed into the model.",
    )
    anomalies_detected = models.PositiveIntegerField(
        default=0,
        help_text="Number of anomalies flagged by this run.",
    )
    alerts_generated = models.PositiveIntegerField(
        default=0,
        help_text="Number of alerts created as a result of this run.",
    )
    status = models.CharField(
        max_length=15,
        choices=Status.choices,
        default=Status.PENDING,
    )
    error_log = models.TextField(
        blank=True,
        default="",
        help_text="Error traceback if the run failed.",
    )

    class Meta:
        db_table = "rakshak_ml_model_run"
        ordering = ["-run_started_at"]
        verbose_name = "ML Model Run"
        verbose_name_plural = "ML Model Runs"
        indexes = [
            models.Index(
                fields=["status", "run_started_at"],
                name="idx_mlrun_status_time",
            ),
        ]

    def __str__(self):
        return f"Run {self.pk} ({self.model.model_name})"


class AnomalyPrediction(TimeStampedModel):
    """
    Per-reading prediction output from an ML model run.

    UniqueConstraint on (model_run, sensor_reading) prevents a run
    from producing duplicate predictions for the same reading.
    """

    model_run = models.ForeignKey(
        MLModelRun,
        on_delete=models.CASCADE,
        related_name="predictions",
        help_text="Parent model run. CASCADE: predictions are deleted with their run.",
    )
    sensor_reading = models.ForeignKey(
        SensorReading,
        on_delete=models.PROTECT,
        related_name="anomaly_predictions",
        help_text="The reading this prediction was made on. PROTECT preserves data.",
    )
    anomaly_score = models.DecimalField(
        max_digits=5,
        decimal_places=4,
        validators=[MinValueValidator(Decimal("0")), MaxValueValidator(Decimal("1"))],
        help_text="Model's anomaly score for this reading, 0.0000–1.0000.",
    )
    is_anomaly = models.BooleanField(
        help_text="Binary classification: True if anomaly_score exceeds model threshold.",
    )
    predicted_fault_type = models.CharField(
        max_length=100,
        blank=True,
        default="",
        help_text="Predicted failure category, e.g. 'rail_fracture', 'bearing_wear'.",
    )
    predicted_failure_in_hours = models.DecimalField(
        max_digits=8,
        decimal_places=2,
        null=True,
        blank=True,
        help_text="Estimated hours until predicted failure occurs.",
    )
    feature_importances = models.JSONField(
        default=dict,
        blank=True,
        help_text=(
            "Feature importance scores from the model. "
            "Example: {'vibration_rms': 0.45, 'temperature_delta': 0.30}."
        ),
    )
    explanation = models.TextField(
        blank=True,
        default="",
        help_text="Human-readable explanation of the prediction.",
    )

    class Meta:
        db_table = "rakshak_anomaly_prediction"
        verbose_name = "Anomaly Prediction"
        verbose_name_plural = "Anomaly Predictions"
        constraints = [
            models.UniqueConstraint(
                fields=["model_run", "sensor_reading"],
                name="uniq_prediction_run_reading",
            ),
        ]

    def __str__(self):
        return f"Pred {self.pk} (score={self.anomaly_score})"


# ===================================================================
# AUDIT LAYER
# ===================================================================

class AuditLog(models.Model):
    """
    Append-only audit trail.

    Does NOT inherit TimeStampedModel — uses its own occurred_at
    timestamp, and must never be updated or deleted.

    Avoids GenericForeignKey/ContentType. Uses plain string fields
    for entity_type and actor_type to remain fully standalone.

    Immutability is enforced at the ORM level via save() and delete()
    overrides. In production, database-level triggers or row-level
    security should supplement this.
    """

    class EventType(models.TextChoices):
        CREATE = "create", "Create"
        UPDATE = "update", "Update"
        DELETE = "delete", "Delete"
        STATUS_CHANGE = "status_change", "Status Change"
        ESCALATION = "escalation", "Escalation"
        LOGIN = "login", "Login"
        SYSTEM = "system", "System"

    class ActorType(models.TextChoices):
        USER = "user", "User"
        SYSTEM = "system", "System"
        ML_PIPELINE = "ml_pipeline", "ML Pipeline"
        SENSOR = "sensor", "Sensor"
        SCHEDULER = "scheduler", "Scheduler"

    event_type = models.CharField(
        max_length=20,
        choices=EventType.choices,
        help_text="Category of the audited event.",
    )
    entity_type = models.CharField(
        max_length=50,
        help_text="Model/table name of the affected entity, e.g. 'alert', 'ticket'.",
    )
    entity_id = models.PositiveBigIntegerField(
        help_text="Primary key of the affected entity row.",
    )
    actor_type = models.CharField(
        max_length=20,
        choices=ActorType.choices,
        help_text="What kind of actor performed the action.",
    )
    actor_identifier = models.CharField(
        max_length=150,
        blank=True,
        default="",
        help_text="Actor identity, e.g. username, service name, sensor_code.",
    )
    previous_state = models.JSONField(
        default=dict,
        blank=True,
        help_text="Snapshot of the entity before the change.",
    )
    new_state = models.JSONField(
        default=dict,
        blank=True,
        help_text="Snapshot of the entity after the change.",
    )
    description = models.TextField(
        blank=True,
        default="",
        help_text="Human-readable summary of the audit event.",
    )
    occurred_at = models.DateTimeField(
        default=timezone.now,
        db_index=True,
        help_text="When the audited event occurred.",
    )

    class Meta:
        db_table = "rakshak_audit_log"
        # No default ordering — high-volume table; let queries specify.
        verbose_name = "Audit Log"
        verbose_name_plural = "Audit Logs"
        indexes = [
            models.Index(
                fields=["entity_type", "entity_id"],
                name="idx_audit_entity",
            ),
        ]

    def __str__(self):
        return f"[{self.event_type}] {self.entity_type}:{self.entity_id}"

    def save(self, *args, **kwargs):
        """Enforce append-only: prevent updates to existing rows."""
        if self.pk is not None:
            raise ValueError(
                "AuditLog entries are immutable. Updates are not permitted. "
                "Create a new entry instead."
            )
        super().save(*args, **kwargs)

    def delete(self, *args, **kwargs):
        """Prevent deletion of audit records."""
        raise ValueError(
            "AuditLog entries cannot be deleted. "
            "This table is an append-only audit trail."
        )
