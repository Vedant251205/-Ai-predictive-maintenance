"""
Central configuration for the AI-Based Predictive Maintenance System.

Every brand string, threshold, machine definition and file path used by the
application is declared here so the whole platform can be re-branded or
re-tuned from a single file.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

DATASET_DIR = BASE_DIR / "dataset"
MODEL_DIR = BASE_DIR / "models"
INSTANCE_DIR = BASE_DIR / "instance"
STATIC_DIR = BASE_DIR / "static"

TELEMETRY_CSV = DATASET_DIR / "machine_telemetry.csv"
KILN_CSV = DATASET_DIR / "kiln_stoppages.csv"
MODEL_PATH = MODEL_DIR / "rf_pipeline.joblib"
MODEL_META_PATH = MODEL_DIR / "model_metadata.json"
# Overridable so a host with a read-only project directory can point the
# database at a writable volume (for example /tmp or a mounted disk).
DATABASE_PATH = Path(
    os.getenv("DATABASE_PATH", str(INSTANCE_DIR / "predictive_maintenance.db"))
)

for _directory in (DATASET_DIR, MODEL_DIR, INSTANCE_DIR):
    _directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Branding  (change these strings to re-brand the entire platform)
# ---------------------------------------------------------------------------
BRAND = {
    "platform_name": os.getenv("PLATFORM_NAME", "AI Maintenance Cockpit"),
    "console_name": "AI Maintenance Console",
    "organisation": "UltraTech Cement",
    "group": "Aditya Birla Group",
    "org_tagline": "PREDICTIVE TELEMETRY",
    "project_title": "AI-Based Predictive Maintenance System",
    "primary_logo_text": "UltraTech",
    "primary_logo_sub": "The Engineer's Choice",
    "secondary_logo_text": "ABG",
    # Drop real artwork at static/img/logo-primary.png / logo-secondary.png and
    # the templates will use the images instead of the CSS wordmarks.
    "primary_logo_file": "img/logo-primary.png",
    "secondary_logo_file": "img/logo-secondary.png",
    # Optional looping backdrop: static/video/plant-loop.mp4
    "background_video_file": "video/plant-loop.mp4",
}


# ---------------------------------------------------------------------------
# Flask
# ---------------------------------------------------------------------------
SECRET_KEY = os.getenv("FLASK_SECRET_KEY", "change-me-in-production")
SESSION_LIFETIME_MINUTES = 240
DEBUG = os.getenv("FLASK_DEBUG", "0") == "1"

# Port supplied by the hosting platform (Render, Railway, Koyeb all set PORT).
PORT = int(os.getenv("PORT", "5000"))

# Public deployment mode. Set APP_ENV=production on the host. This tightens the
# session cookie, hides the demo-credential helper on the login page, and makes
# the platform refuse to start on the shipped placeholder secret key.
APP_ENV = os.getenv("APP_ENV", "development").strip().lower()
IS_PRODUCTION = APP_ENV == "production"

# The demo access-key button autofills working credentials. Harmless locally,
# an open invitation on a public URL.
SHOW_DEMO_KEYS = os.getenv(
    "SHOW_DEMO_KEYS", "0" if IS_PRODUCTION else "1"
) == "1"


def production_warnings() -> list[str]:
    """Configuration problems that matter only on a public deployment."""
    problems = []
    if SECRET_KEY == "change-me-in-production":
        problems.append(
            "FLASK_SECRET_KEY is still the shipped placeholder. Sessions can be "
            "forged. Set a long random value in the host's environment."
        )
    if ADMIN_PASSWORD == "admin123":
        problems.append(
            "The administrator password is still the demo value 'admin123'. "
            "Set ADMIN_PASSWORD in the host's environment."
        )
    if EMPLOYEE_PASSWORD == "employee123":
        problems.append(
            "The employee password is still the demo value 'employee123'. "
            "Set EMPLOYEE_PASSWORD in the host's environment."
        )
    return problems


# ---------------------------------------------------------------------------
# Notification gateways.  Both stay disabled until real credentials exist,
# so the platform runs end-to-end with zero third-party configuration.
# ---------------------------------------------------------------------------
EMAIL = {
    "enabled": os.getenv("EMAIL_ENABLED", "0") == "1",
    "sender": os.getenv("PLATFORM_EMAIL", ""),
    "password": os.getenv("PLATFORM_EMAIL_PASSWORD", ""),
    "host": os.getenv("SMTP_HOST", "smtp.gmail.com"),
    "port": int(os.getenv("SMTP_PORT", "587")),
    "display_name": os.getenv(
        "EMAIL_DISPLAY_NAME", "UltraTech Cement AI Maintenance System"
    ),
}

SMS = {
    "enabled": os.getenv("SMS_ENABLED", "0") == "1",
    "account_sid": os.getenv("TWILIO_SID", ""),
    "auth_token": os.getenv("TWILIO_TOKEN", ""),
    "from_number": os.getenv("TWILIO_FROM", ""),
}


# ---------------------------------------------------------------------------
# Machine learning contract
# ---------------------------------------------------------------------------
ML = {
    "n_estimators": 150,
    "max_depth": 16,
    "min_samples_leaf": 1,
    "random_state": 42,
    "test_size": 0.25,
}

# Numeric sensor inputs, in the exact order the pipeline expects.
NUMERIC_FEATURES = [
    "air_temperature_k",
    "process_temperature_k",
    "rotational_speed_rpm",
    "torque_nm",
    "tool_wear_min",
]
CATEGORICAL_FEATURES = ["machine_type"]
TARGET = "machine_failure"

FEATURE_LABELS = {
    "air_temperature_k": "Air Temperature (K)",
    "process_temperature_k": "Process Temp (K)",
    "rotational_speed_rpm": "Rotational Speed (RPM)",
    "torque_nm": "Torque (Nm)",
    "tool_wear_min": "Tool Wear (min)",
    "machine_type": "Machine Type / Quality",
}


# ---------------------------------------------------------------------------
# Synthetic dataset generation.  Short on purpose, but the physical
# relationships (collinearity) between sensors are preserved.
# ---------------------------------------------------------------------------
DATASET = {
    "rows": 1500,
    "random_seed": 42,
    # Process temperature tracks air temperature with a ~10 K offset: this is
    # the strongest collinear pair in the AI4I 2020 reference data.
    "air_temp_mean": 300.0,
    "air_temp_sd": 2.0,
    "process_offset_mean": 10.0,
    "process_offset_sd": 1.0,
    # Rotational speed and torque are coupled through mechanical power
    # (P = torque * omega), which produces the characteristic inverse
    # correlation between the two columns.
    "target_power_w_mean": 2860.0,
    "target_power_w_sd": 160.0,
    "torque_mean": 40.0,
    "torque_sd": 10.0,
    "tool_wear_max": 253,
    "type_mix": {"L": 0.60, "M": 0.30, "H": 0.10},
}

# Operating envelopes shown to the operator on the cockpit page.
# min/max are the hard input limits; normal_min/normal_max drive the green
# "healthy band" indicator underneath each slider.
OPERATING_RANGES = {
    "air_temperature_k": {
        "min": 290.0, "max": 315.0, "default": 300.0, "step": 0.1,
        "normal_min": 295.0, "normal_max": 305.0, "unit": "K",
        "icon": "fa-temperature-half", "colour": "danger",
        "note": "Safe ambient threshold: < 305 K. High ambient heat reduces "
                "heat dissipation.",
    },
    "process_temperature_k": {
        "min": 300.0, "max": 332.0, "default": 310.0, "step": 0.1,
        "normal_min": 305.0, "normal_max": 315.0, "unit": "K",
        "icon": "fa-thermometer", "colour": "warning",
        "note": "Process temp difference (dT) must stay above 8.6 K, otherwise "
                "the unit cannot shed heat.",
    },
    "rotational_speed_rpm": {
        "min": 1000, "max": 3000, "default": 1500, "step": 1,
        "normal_min": 1300, "normal_max": 1700, "unit": "RPM",
        "icon": "fa-rotate", "colour": "info",
        "note": "Rated envelope 1300-1700 RPM. Above 2570 RPM bearing "
                "overspeed becomes likely.",
    },
    "torque_nm": {
        "min": 3.0, "max": 80.0, "default": 40.0, "step": 0.1,
        "normal_min": 30.0, "normal_max": 50.0, "unit": "Nm",
        "icon": "fa-bolt", "colour": "primary",
        "note": "Continuous rating 30-50 Nm. Sustained load above 56 Nm "
                "overloads the drive train.",
    },
    "tool_wear_min": {
        "min": 0, "max": 260, "default": 0, "step": 1,
        "normal_min": 0, "normal_max": 200, "unit": "min",
        "icon": "fa-screwdriver-wrench", "colour": "success",
        "note": "Replace wear parts before 200 min. Overstrain risk rises "
                "sharply when combined with high torque.",
    },
}

MACHINE_TYPES = {
    "L": "L (Low / Light-duty)",
    "M": "M (Medium / Standard-duty)",
    "H": "H (High / Heavy-duty)",
}


# ---------------------------------------------------------------------------
# Health scoring, risk banding and Remaining Useful Life
# ---------------------------------------------------------------------------
# Health score is the direct complement of the model's failure probability.
HEALTH_BANDS = [
    # (inclusive lower bound, status, colour token, recommended action)
    (85.0, "Excellent", "success", "Monitor"),
    (65.0, "Good", "info", "Schedule Inspection"),
    (45.0, "Warning", "warning", "Plan Maintenance"),
    (0.0, "Critical", "danger", "Immediate Shutdown"),
]

RUL = {
    # Baseline service life of a healthy asset, in operating hours.
    "base_life_hours": 7600.0,
    "tool_wear_max": 253.0,
    # Wear term: how much of the baseline life a fully worn tool consumes.
    "wear_weight": 0.55,
    # Health term floor: even a failing unit retains some residual life.
    "health_floor": 0.45,
    # Stress term: deviation from the nominal operating point.
    "thermal_nominal_delta": 10.0,
    "thermal_weight": 0.20,
    "thermal_low_multiplier": 1.5,
    "speed_nominal_rpm": 1700.0,
    "speed_span_rpm": 1500.0,
    "speed_weight": 0.15,
    "torque_nominal_nm": 50.0,
    "torque_span_nm": 40.0,
    "torque_weight": 0.15,
    "stress_floor": 0.35,
}

NEXT_SERVICE_DAYS = {
    "Excellent": 90,
    "Good": 60,
    "Warning": 30,
    "Critical": 7,
}

# Alert thresholds used by the Alert Center.
ALERT_THRESHOLDS = {
    "critical_health": 45.0,
    "warning_health": 65.0,
    "critical_failure_prob": 55.0,
    "warning_failure_prob": 30.0,
    "critical_tool_wear": 200,
    "min_thermal_delta": 8.6,
}

SEVERITY_OPTIONS = [
    ("critical", "Critical & Failures Only"),
    ("warning", "Warning and above"),
    ("all", "All Alerts"),
]


# ---------------------------------------------------------------------------
# CMMS work-order priorities
# ---------------------------------------------------------------------------
PRIORITIES = {
    "P1": {"label": "Emergency", "colour": "danger", "man_hours": 6.0,
           "sla": "Immediate", "order": 1},
    "P2": {"label": "Urgent", "colour": "warning", "man_hours": 4.0,
           "sla": "Within 24 hours", "order": 2},
    "P3": {"label": "High", "colour": "amber", "man_hours": 2.5,
           "sla": "Within 7 days", "order": 3},
    "P4": {"label": "Scheduled", "colour": "info", "man_hours": 1.5,
           "sla": "Next shutdown window", "order": 4},
    "P5": {"label": "Preventive", "colour": "success", "man_hours": 0.5,
           "sla": "Routine round", "order": 5},
}

DEPARTMENTS = ["Mechanical", "Electrical", "Instrumentation"]


# ---------------------------------------------------------------------------
# Monitored fleet
# ---------------------------------------------------------------------------
MACHINES = [
    {"id": "KLN-01", "name": "Rotary Kiln Line 1", "type": "H",
     "department": "Mechanical", "category": "Pyro Process", "icon": "fa-fire"},
    {"id": "KLN-02", "name": "Rotary Kiln Line 2", "type": "H",
     "department": "Mechanical", "category": "Pyro Process", "icon": "fa-fire"},
    {"id": "BLR-03", "name": "Bag Filter Blower Unit", "type": "M",
     "department": "Electrical", "category": "Air Handling", "icon": "fa-wind"},
    {"id": "RMM-04", "name": "Raw Material Ball Mill", "type": "H",
     "department": "Mechanical", "category": "Grinding", "icon": "fa-gears"},
    {"id": "CML-05", "name": "Coal Mill Drive", "type": "M",
     "department": "Mechanical", "category": "Grinding", "icon": "fa-industry"},
    {"id": "CLC-06", "name": "Clinker Cooler Fan", "type": "M",
     "department": "Instrumentation", "category": "Cooling", "icon": "fa-snowflake"},
    {"id": "FAN-07", "name": "Preheater Fan", "type": "M",
     "department": "Electrical", "category": "Air Handling", "icon": "fa-fan"},
    {"id": "CRS-08", "name": "Limestone Crusher", "type": "L",
     "department": "Mechanical", "category": "Crushing", "icon": "fa-hammer"},
    {"id": "PKR-09", "name": "Cement Packing Line", "type": "L",
     "department": "Instrumentation", "category": "Packing", "icon": "fa-boxes-stacked"},
]

MACHINE_INDEX = {machine["id"]: machine for machine in MACHINES}


# ---------------------------------------------------------------------------
# Kiln stoppage analytics dataset shape
# ---------------------------------------------------------------------------
KILN = {
    "unit": "U1KILN",
    "start_date": "2025-05-04",
    "end_date": "2026-03-31",
    "days": 332,
    "target_stoppages": 51,
    "target_downtime_hours": 659.7,
    "random_seed": 7,
    "causes": [
        "Refractory / Coating Failure",
        "Mechanical Breakdown",
        "Electrical Fault",
        "Instrumentation Fault",
        "Fan / Blower Trip",
        "Process Upset",
        "Raw Material Shortage",
        "Planned Preventive Halt",
    ],
}


# ---------------------------------------------------------------------------
# Demo accounts seeded on first run.  Generic placeholder identities only.
# ---------------------------------------------------------------------------
# Seeded credentials. Overridable from the environment so a public deployment
# never has to run on the documented demo passwords.
ADMIN_EMAIL = os.getenv("ADMIN_EMAIL", "admin@ultratech.com")
ADMIN_USER_ID = os.getenv("ADMIN_USER_ID", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "admin123")

EMPLOYEE_EMAIL = os.getenv("EMPLOYEE_EMAIL", "employee@ultratech.com")
EMPLOYEE_USER_ID = os.getenv("EMPLOYEE_USER_ID", "employee")
EMPLOYEE_PASSWORD = os.getenv("EMPLOYEE_PASSWORD", "employee123")

SEED_USERS = [
    {"user_id": ADMIN_USER_ID, "name": "Admin Plant Manager",
     "email": ADMIN_EMAIL, "password": ADMIN_PASSWORD,
     "role": "admin", "department": "Management"},
    {"user_id": EMPLOYEE_USER_ID, "name": "Shift Engineer",
     "email": EMPLOYEE_EMAIL, "password": EMPLOYEE_PASSWORD,
     "role": "employee", "department": "Operations"},
    {"user_id": "UT-1042", "name": "Maintenance Planner",
     "email": "planner@ultratech.com", "password": "planner123",
     "role": "employee", "department": "Maintenance"},
    {"user_id": "UT-1043", "name": "Instrumentation Lead",
     "email": "instrument@ultratech.com", "password": "instrument123",
     "role": "employee", "department": "Instrumentation"},
]

DEMO_KEYS = {
    "admin": {"email": ADMIN_EMAIL, "user_id": ADMIN_USER_ID,
              "password": ADMIN_PASSWORD},
    "employee": {"email": EMPLOYEE_EMAIL, "user_id": EMPLOYEE_USER_ID,
                 "password": EMPLOYEE_PASSWORD},
}

ROLE_OPTIONS = [
    ("employee", "Employee (Full Access)"),
    ("admin", "Administrator (Root Privileges)"),
]

# Navigation model consumed by the sidebar partial.
NAV_SECTIONS = [
    {
        "label": "Operations",
        "icon": "fa-chart-simple",
        "items": [
            {"endpoint": "main.dashboard", "label": "Dashboard", "icon": "fa-gauge-high"},
            {"endpoint": "main.alerts", "label": "Alert Center", "icon": "fa-bell",
             "badge": "alert_count"},
            {"endpoint": "main.executive_dashboard", "label": "Executive Dashboard",
             "icon": "fa-chart-pie"},
        ],
    },
    {
        "label": "Analytics",
        "icon": "fa-chart-line",
        "items": [
            {"endpoint": "main.analytics", "label": "Analytics", "icon": "fa-chart-line"},
            {"endpoint": "main.kiln_analytics", "label": "Kiln Stoppages", "icon": "fa-fire"},
            {"endpoint": "main.feature_intelligence", "label": "AI Feature Intelligence",
             "icon": "fa-diagram-project"},
        ],
    },
    {
        "label": "Maintenance",
        "icon": "fa-screwdriver-wrench",
        "items": [
            {"endpoint": "main.maintenance_advisor", "label": "Maintenance Advisor",
             "icon": "fa-screwdriver-wrench"},
            {"endpoint": "predict.cockpit", "label": "Predictive Cockpit",
             "icon": "fa-robot"},
        ],
    },
    {
        "label": "System",
        "icon": "fa-server",
        "items": [
            {"endpoint": "history.records", "label": "Data History", "icon": "fa-database"},
            {"endpoint": "main.architecture", "label": "Architecture", "icon": "fa-sitemap"},
            {"endpoint": "main.future_roadmap", "label": "Future Roadmap",
             "icon": "fa-rocket"},
        ],
    },
]

TECH_STACK = [
    {"label": "Python 3.13", "icon": "fa-python", "colour": "info"},
    {"label": "Flask 3.1", "icon": "fa-flask", "colour": "success"},
    {"label": "Random Forest ML", "icon": "fa-robot", "colour": "danger"},
    {"label": "SQLite", "icon": "fa-database", "colour": "warning"},
    {"label": f"{len(MACHINES)} Machines Monitored", "icon": "fa-industry",
     "colour": "primary"},
]
