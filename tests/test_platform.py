"""
End-to-end test suite for the AI Predictive Maintenance System.

Written against the standard library `unittest` rather than pytest, so it runs
on a bare Python install with no extra packages:

    python -m unittest discover -s tests -v          (from the project root)
    python -m unittest tests.test_platform -v        (this module only)

The suite points the application at a throwaway SQLite file, so running it
never touches instance/predictive_maintenance.db.
"""

from __future__ import annotations

import csv
import io
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config                                            # noqa: E402
import pandas as pd                                      # noqa: E402

_REAL_DATABASE_PATH = config.DATABASE_PATH
_TEMP_DIRECTORY: tempfile.TemporaryDirectory | None = None

TOKEN_PATTERN = re.compile(r'name="csrf-token" content="([^"]+)"')

PAGES = [
    "/dashboard/admin",
    "/dashboard/employee",
    "/alerts",
    "/executive-dashboard",
    "/analytics",
    "/feature-intelligence",
    "/kiln-analytics",
    "/maintenance-advisor",
    "/predict",
    "/history",
    "/architecture",
    "/future-roadmap",
]

API_GET_ENDPOINTS = [
    "/api/machines",
    "/api/fleet-status",
    "/api/predictions",
    "/api/alerts",
    "/api/kpis",
    "/api/kiln-stats",
    "/api/feature-importance",
]


def setUpModule() -> None:
    """Redirect persistence to a temporary database, then build everything."""
    global _TEMP_DIRECTORY

    _TEMP_DIRECTORY = tempfile.TemporaryDirectory(prefix="pdm-tests-")
    config.DATABASE_PATH = Path(_TEMP_DIRECTORY.name) / "test.db"

    from services import database, ml_service

    # Only regenerates artefacts that are actually missing.
    ml_service.bootstrap(verbose=False)
    database.init_db()


def tearDownModule() -> None:
    config.DATABASE_PATH = _REAL_DATABASE_PATH
    if _TEMP_DIRECTORY is not None:
        try:
            _TEMP_DIRECTORY.cleanup()
        except (OSError, PermissionError):
            pass


class ClientCase(unittest.TestCase):
    """Base case providing an authenticated Flask test client."""

    @classmethod
    def setUpClass(cls) -> None:
        import app as application

        cls.application = application
        cls.client = application.app.test_client()

    def csrf(self, path: str = "/login") -> str:
        page = self.client.get(path, follow_redirects=True)
        found = TOKEN_PATTERN.search(page.get_data(as_text=True))
        return found.group(1) if found else ""

    def sign_in(self, role: str = "admin") -> None:
        """Sign in, then re-read the token: login rotates the session."""
        self.client.get("/logout")
        credentials = config.DEMO_KEYS[role]
        self.client.post("/login", data={
            "csrf_token": self.csrf(),
            "role": role,
            "email": credentials["email"],
            "user_id": credentials["user_id"],
            "password": credentials["password"],
        })

    def token_after_login(self) -> str:
        return self.csrf("/predict")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class TestSyntheticDataset(unittest.TestCase):
    """The dataset must stay short and keep its physical collinearity."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.frame = pd.read_csv(config.TELEMETRY_CSV)

    def test_dataset_is_short(self):
        self.assertLessEqual(len(self.frame), 2000,
                             "dataset should stay small and readable")
        self.assertGreaterEqual(len(self.frame), 800,
                                "dataset needs enough rows to train on")

    def test_required_columns_present(self):
        for column in config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES:
            self.assertIn(column, self.frame.columns)
        self.assertIn(config.TARGET, self.frame.columns)

    def test_failure_prevalence_is_learnable(self):
        rate = self.frame[config.TARGET].mean()
        self.assertGreater(rate, 0.10, "too few failures to learn from")
        self.assertLess(rate, 0.50, "failures should stay the minority class")

    def test_thermal_channels_are_positively_collinear(self):
        r = self.frame["air_temperature_k"].corr(
            self.frame["process_temperature_k"])
        self.assertGreater(
            r, 0.75,
            f"ambient heat must propagate into the process (r={r:.3f})")

    def test_torque_and_speed_are_inversely_collinear(self):
        r = self.frame["torque_nm"].corr(self.frame["rotational_speed_rpm"])
        self.assertLess(
            r, -0.65,
            f"torque and speed share delivered power (r={r:.3f})")

    def test_mechanical_power_is_consistent(self):
        """Speed really is derived from power, not drawn independently."""
        from utils.recommendations import mechanical_power_w

        computed = self.frame.apply(
            lambda row: mechanical_power_w(
                row["torque_nm"], row["rotational_speed_rpm"]),
            axis=1,
        )
        difference = (computed - self.frame["power_w"]).abs().max()
        # The CSV stores speed as an integer and torque to one decimal, so
        # recomputing from the published columns carries rounding error of up to
        # roughly (torque x 0.5 + rpm x 0.05) x 2pi/60, about 20 W at the top of
        # the range.
        self.assertLess(difference, 25.0,
                        "power_w column must match torque x angular velocity")

    def test_variance_inflation_shows_shared_information(self):
        sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
        from generate_synthetic_dataset import variance_inflation_factors

        vif = variance_inflation_factors(self.frame, config.NUMERIC_FEATURES)
        coupled = [
            vif["air_temperature_k"],
            vif["process_temperature_k"],
            vif["rotational_speed_rpm"],
            vif["torque_nm"],
        ]
        for value in coupled:
            self.assertGreater(
                value, 1.5,
                "coupled channels should retain shared information")
            self.assertLess(
                value, 40.0,
                "collinearity should be realistic, not degenerate")

    def test_hazard_labels_are_graded_not_binary(self):
        """Graded hazards are what keep the middle risk bands reachable."""
        self.assertIn("hazard", self.frame.columns)
        middle = self.frame["hazard"].between(0.15, 0.85).mean()
        self.assertGreater(
            middle, 0.10,
            "a meaningful share of readings must carry intermediate risk")

    def test_hazard_is_calibrated_against_observed_failures(self):
        low = self.frame[self.frame["hazard"] < 0.2][config.TARGET].mean()
        high = self.frame[self.frame["hazard"] > 0.8][config.TARGET].mean()
        self.assertLess(low, 0.20)
        self.assertGreater(high, 0.80)
        self.assertGreater(high, low)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class TestModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from services import ml_service

        cls.ml = ml_service
        cls.metrics = ml_service.metrics()
        cls.importance = ml_service.feature_importance()

    def test_artefacts_exist(self):
        self.assertTrue(config.MODEL_PATH.exists())
        self.assertTrue(config.MODEL_META_PATH.exists())

    def test_model_card_matches_declared_architecture(self):
        card = self.ml.model_card()
        self.assertEqual(card["algorithm"], "RandomForestClassifier")
        self.assertEqual(card["n_estimators"], 150)
        self.assertIn("StandardScaler", card["preprocessor"])
        self.assertIn("OneHotEncoder", card["preprocessor"])

    def test_metrics_are_credible(self):
        for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
            self.assertIn(key, self.metrics)
            self.assertGreater(self.metrics[key], 55.0,
                               f"{key} is implausibly low")
            self.assertLessEqual(self.metrics[key], 100.0)
        self.assertGreater(self.metrics["roc_auc"], 75.0)

    def test_importances_sum_to_one_hundred(self):
        total = sum(entry["pct"] for entry in self.importance)
        self.assertAlmostEqual(total, 100.0, places=0)

    def test_mechanical_load_dominates_importance(self):
        ranked = {entry["feature"]: entry["pct"] for entry in self.importance}
        self.assertGreater(ranked["torque_nm"], ranked["tool_wear_min"])
        self.assertGreater(ranked["rotational_speed_rpm"],
                           ranked["tool_wear_min"])
        top_three = (ranked["torque_nm"]
                     + ranked["rotational_speed_rpm"]
                     + ranked["tool_wear_min"])
        self.assertGreater(top_three, 60.0)

    def test_probability_is_bounded(self):
        reading = {
            "air_temperature_k": 300.0, "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500, "torque_nm": 40.0,
            "tool_wear_min": 0, "machine_type": "M",
        }
        value = self.ml.failure_probability(reading)
        self.assertGreaterEqual(value, 0.0)
        self.assertLessEqual(value, 100.0)

    def test_healthy_reading_scores_better_than_overloaded_one(self):
        healthy = {
            "air_temperature_k": 299.0, "process_temperature_k": 309.5,
            "rotational_speed_rpm": 1480, "torque_nm": 38.0,
            "tool_wear_min": 15, "machine_type": "M",
        }
        overloaded = {
            "air_temperature_k": 303.0, "process_temperature_k": 311.5,
            "rotational_speed_rpm": 1420, "torque_nm": 68.0,
            "tool_wear_min": 225, "machine_type": "L",
        }
        self.assertLess(self.ml.failure_probability(healthy),
                        self.ml.failure_probability(overloaded))

    def test_batch_and_single_inference_agree(self):
        readings = [
            {"air_temperature_k": 300.0, "process_temperature_k": 310.0,
             "rotational_speed_rpm": 1500, "torque_nm": 40.0,
             "tool_wear_min": 10, "machine_type": "M"},
            {"air_temperature_k": 302.0, "process_temperature_k": 311.0,
             "rotational_speed_rpm": 1600, "torque_nm": 55.0,
             "tool_wear_min": 120, "machine_type": "H"},
        ]
        batch = self.ml.failure_probabilities(readings)
        singles = [self.ml.failure_probability(item) for item in readings]
        self.assertEqual(batch, singles)

    def test_collinearity_audit_is_published(self):
        audit = self.ml.collinearity()
        self.assertTrue(audit.get("vif"))
        self.assertTrue(audit.get("correlation_matrix"))
        self.assertEqual(len(audit["preserved_relationships"]), 3)


# ---------------------------------------------------------------------------
# Prediction service
# ---------------------------------------------------------------------------
class TestPredictionService(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from services import prediction_service

        cls.service = prediction_service

    def test_health_is_the_complement_of_failure_probability(self):
        result = self.service.assess({
            "air_temperature_k": 301.0, "process_temperature_k": 311.5,
            "rotational_speed_rpm": 1550, "torque_nm": 44.0,
            "tool_wear_min": 60, "machine_type": "M",
        })
        self.assertAlmostEqual(
            result["health_score"], 100.0 - result["failure_prob"], places=1)

    def test_bands_map_to_the_documented_thresholds(self):
        self.assertEqual(self.service.health_band(97.0)[0], "Excellent")
        self.assertEqual(self.service.health_band(85.0)[0], "Excellent")
        self.assertEqual(self.service.health_band(70.0)[0], "Good")
        self.assertEqual(self.service.health_band(55.0)[0], "Warning")
        self.assertEqual(self.service.health_band(20.0)[0], "Critical")

    def test_rul_falls_as_tool_wear_rises(self):
        base = {
            "air_temperature_k": 300.0, "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500, "torque_nm": 40.0,
            "tool_wear_min": 0, "machine_type": "M",
        }
        fresh = self.service.remaining_useful_life(base, 5.0)["hours"]
        worn = self.service.remaining_useful_life(
            {**base, "tool_wear_min": 240}, 5.0)["hours"]
        self.assertLess(worn, fresh)

    def test_rul_falls_as_failure_probability_rises(self):
        base = {
            "air_temperature_k": 300.0, "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500, "torque_nm": 40.0,
            "tool_wear_min": 50, "machine_type": "M",
        }
        healthy = self.service.remaining_useful_life(base, 2.0)["hours"]
        failing = self.service.remaining_useful_life(base, 90.0)["hours"]
        self.assertLess(failing, healthy)

    def test_rul_terms_stay_inside_their_bounds(self):
        result = self.service.remaining_useful_life({
            "air_temperature_k": 305.0, "process_temperature_k": 313.0,
            "rotational_speed_rpm": 2800, "torque_nm": 75.0,
            "tool_wear_min": 250, "machine_type": "L",
        }, 95.0)
        self.assertGreaterEqual(result["terms"]["stress"],
                                config.RUL["stress_floor"] - 1e-9)
        self.assertGreater(result["hours"], 0)

    def test_next_service_never_exceeds_its_band_ceiling(self):
        for preset in self.service.PRESETS.values():
            result = self.service.assess(dict(preset["values"]))
            ceiling = config.NEXT_SERVICE_DAYS[result["status"]]
            self.assertLessEqual(result["next_service_days"], ceiling)
            self.assertGreaterEqual(result["next_service_days"], 1)

    def test_critical_preset_is_worse_than_healthy_preset(self):
        healthy = self.service.assess(
            dict(self.service.PRESETS["healthy"]["values"]))
        critical = self.service.assess(
            dict(self.service.PRESETS["critical"]["values"]))
        self.assertGreater(healthy["health_score"], critical["health_score"])
        self.assertEqual(healthy["status"], "Excellent")
        self.assertIn(critical["status"], ("Warning", "Critical"))

    def test_assessment_exposes_everything_the_ui_renders(self):
        result = self.service.assess({
            "air_temperature_k": 300.0, "process_temperature_k": 310.0,
            "rotational_speed_rpm": 1500, "torque_nm": 40.0,
            "tool_wear_min": 20, "machine_type": "M",
            "machine_id": "KLN-01",
        })
        for key in ("health_score", "failure_prob", "status", "colour",
                    "rul_hours", "rul_days", "next_service_days", "action",
                    "priority", "priority_label", "drivers",
                    "recommendation", "work_order", "thermal_delta_k",
                    "power_w", "machine_name"):
            self.assertIn(key, result)
        self.assertEqual(result["machine_name"], "Rotary Kiln Line 1")

    def test_overloaded_reading_raises_a_critical_driver(self):
        result = self.service.assess({
            "air_temperature_k": 303.0, "process_temperature_k": 311.0,
            "rotational_speed_rpm": 1400, "torque_nm": 72.0,
            "tool_wear_min": 230, "machine_type": "L",
        })
        severities = {driver["severity"] for driver in result["drivers"]}
        self.assertIn("critical", severities)


# ---------------------------------------------------------------------------
# Fleet simulator
# ---------------------------------------------------------------------------
class TestFleetService(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from services import fleet_service

        cls.fleet = fleet_service

    def test_every_configured_machine_is_reported(self):
        snapshot = self.fleet.snapshot(bucket=1)
        self.assertEqual(len(snapshot), len(config.MACHINES))
        self.assertEqual(
            {item["machine_id"] for item in snapshot},
            {machine["id"] for machine in config.MACHINES},
        )

    def test_snapshot_is_deterministic_within_a_window(self):
        first = self.fleet.snapshot(bucket=99)
        second = self.fleet.snapshot(bucket=99)
        self.assertEqual(
            [item["health_score"] for item in first],
            [item["health_score"] for item in second],
        )

    def test_fleet_mix_is_stable_across_windows(self):
        """Five healthy, two warning, two critical, every window."""
        for bucket in range(1, 11):
            summary = self.fleet.summary(self.fleet.snapshot(bucket=bucket))
            self.assertEqual(summary["healthy"], 5, f"bucket {bucket}")
            self.assertEqual(summary["warning"], 2, f"bucket {bucket}")
            self.assertEqual(summary["critical"], 2, f"bucket {bucket}")

    def test_priority_mix_matches_the_documented_schedule(self):
        summary = self.fleet.summary(self.fleet.snapshot(bucket=1))
        counts = summary["priority_counts"]
        self.assertEqual(counts["P2"], 2)
        self.assertEqual(counts["P3"], 2)
        self.assertEqual(counts["P5"], 5)

    def test_mechanical_power_stays_inside_the_envelope(self):
        """No duty profile may sit outside the plant power band."""
        for bucket in range(1, 6):
            for machine in self.fleet.snapshot(bucket=bucket):
                self.assertGreater(machine["power_w"], 3000.0,
                                   machine["machine_id"])
                self.assertLess(machine["power_w"], 9600.0,
                                machine["machine_id"])

    def test_summary_totals_are_internally_consistent(self):
        snapshot = self.fleet.snapshot(bucket=3)
        summary = self.fleet.summary(snapshot)
        self.assertEqual(summary["healthy"] + summary["non_healthy"],
                         summary["total"])
        self.assertEqual(summary["excellent"] + summary["good"],
                         summary["healthy"])
        self.assertEqual(sum(summary["priority_counts"].values()),
                         summary["total"])

    def test_health_trend_returns_matching_series(self):
        trend = self.fleet.health_trend(6)
        self.assertEqual(len(trend["labels"]), 6)
        self.assertEqual(len(trend["values"]), 6)
        for value in trend["values"]:
            self.assertGreaterEqual(value, 0.0)
            self.assertLessEqual(value, 100.0)

    def test_work_orders_are_sorted_worst_first(self):
        orders = self.fleet.work_orders(self.fleet.snapshot(bucket=1))
        ranks = [config.PRIORITIES[item["priority"]]["order"]
                 for item in orders]
        self.assertEqual(ranks, sorted(ranks))

    def test_lookup_by_machine_id(self):
        self.assertIsNotNone(self.fleet.machine("KLN-01"))
        self.assertIsNone(self.fleet.machine("NOPE-99"))


# ---------------------------------------------------------------------------
# Kiln analytics
# ---------------------------------------------------------------------------
class TestKilnAnalytics(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from services import kiln_service

        cls.service = kiln_service
        cls.kpis = kiln_service.kpis()

    def test_reported_figures_match_the_specification(self):
        self.assertEqual(self.kpis["days"], 332)
        self.assertEqual(self.kpis["stoppages"], 51)
        self.assertAlmostEqual(self.kpis["downtime_hours"], 659.7, places=1)
        self.assertAlmostEqual(self.kpis["availability"], 91.7, places=1)
        self.assertAlmostEqual(self.kpis["mtbf"], 156.2, places=1)
        self.assertAlmostEqual(self.kpis["mttr"], 12.9, places=1)

    def test_uptime_and_downtime_account_for_the_whole_window(self):
        self.assertAlmostEqual(
            self.kpis["uptime_hours"] + self.kpis["downtime_hours"],
            self.kpis["total_hours"],
            places=1,
        )

    def test_mttr_is_downtime_over_stoppages(self):
        self.assertAlmostEqual(
            self.kpis["mttr"],
            round(self.kpis["downtime_hours"] / self.kpis["stoppages"], 1),
            places=1,
        )

    def test_planned_and_unplanned_split_is_complete(self):
        self.assertEqual(
            self.kpis["planned_count"] + self.kpis["unplanned_count"],
            self.kpis["stoppages"],
        )

    def test_cause_pareto_is_ordered_and_totals_one_hundred(self):
        causes = self.service.cause_breakdown()
        hours = [row["hours"] for row in causes]
        self.assertEqual(hours, sorted(hours, reverse=True))
        self.assertAlmostEqual(causes[-1]["cumulative"], 100.0, places=0)
        self.assertEqual(sum(row["events"] for row in causes),
                         self.kpis["stoppages"])

    def test_monthly_series_lines_up(self):
        monthly = self.service.monthly_trend()
        self.assertEqual(len(monthly["labels"]), len(monthly["hours"]))
        self.assertEqual(len(monthly["labels"]), len(monthly["events"]))
        self.assertEqual(sum(monthly["events"]), self.kpis["stoppages"])

    def test_duration_histogram_covers_every_stoppage(self):
        histogram = self.service.duration_histogram()
        self.assertEqual(sum(histogram["values"]), self.kpis["stoppages"])

    def test_stoppages_never_overlap(self):
        frame = self.service.load().sort_values("start_dt")
        ends = frame["end_dt"].tolist()
        starts = frame["start_dt"].tolist()
        for index in range(1, len(starts)):
            self.assertLessEqual(
                ends[index - 1], starts[index],
                "a stoppage must finish before the next one begins")


# ---------------------------------------------------------------------------
# Validators
# ---------------------------------------------------------------------------
class TestValidators(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from utils import validators

        cls.validators = validators

    def _sensor_form(self, **overrides) -> dict:
        form = {
            "air_temperature_k": "300.0",
            "process_temperature_k": "310.0",
            "rotational_speed_rpm": "1500",
            "torque_nm": "40.0",
            "tool_wear_min": "20",
            "machine_type": "M",
            "machine_id": "",
            "runtime_hours": "1000",
        }
        form.update(overrides)
        return form

    def test_valid_reading_passes(self):
        cleaned, errors = self.validators.validate_sensor_form(
            self._sensor_form())
        self.assertEqual(errors, [])
        self.assertEqual(cleaned["rotational_speed_rpm"], 1500)

    def test_out_of_range_value_is_rejected(self):
        _, errors = self.validators.validate_sensor_form(
            self._sensor_form(torque_nm="500"))
        self.assertTrue(errors)

    def test_non_numeric_value_is_rejected(self):
        _, errors = self.validators.validate_sensor_form(
            self._sensor_form(torque_nm="abc"))
        self.assertTrue(errors)

    def test_unknown_machine_type_is_rejected(self):
        _, errors = self.validators.validate_sensor_form(
            self._sensor_form(machine_type="Z"))
        self.assertTrue(errors)

    def test_unknown_machine_id_is_rejected(self):
        _, errors = self.validators.validate_sensor_form(
            self._sensor_form(machine_id="XXX-99"))
        self.assertTrue(errors)

    def test_employee_form_rejects_bad_input(self):
        _, errors = self.validators.validate_employee_form(
            {"user_id": "a", "name": "x", "email": "not-an-email",
             "password": "123", "role": "wizard"},
            existing_ids=set(),
        )
        self.assertGreaterEqual(len(errors), 4)

    def test_employee_form_rejects_duplicate_id(self):
        _, errors = self.validators.validate_employee_form(
            {"user_id": "admin", "name": "Someone New",
             "email": "new@ultratech.com", "password": "secret1",
             "role": "employee", "department": "Operations"},
            existing_ids={"admin"},
        )
        self.assertTrue(any("already exists" in message
                            for message in errors))

    def test_alert_settings_require_a_recipient_when_enabled(self):
        _, errors = self.validators.validate_alert_settings(
            {"email_enabled": "on", "recipient_email": "",
             "severity": "critical"})
        self.assertTrue(errors)


# ---------------------------------------------------------------------------
# Authentication and access control
# ---------------------------------------------------------------------------
class TestAuthentication(ClientCase):

    def setUp(self) -> None:
        # The client is shared across the class, and unittest runs methods in
        # alphabetical order, so an earlier test can leave a live session
        # behind. Start every case signed out.
        self.client.get("/logout")

    def test_anonymous_access_redirects_to_login(self):
        for path in ["/dashboard", "/predict", "/history", "/alerts",
                     "/api/kpis"]:
            response = self.client.get(path)
            self.assertEqual(response.status_code, 302, path)
            self.assertIn("/login", response.headers["Location"], path)

    def test_valid_credentials_sign_in(self):
        self.client.get("/logout")
        response = self.client.post("/login", data={
            "csrf_token": self.csrf(), "role": "admin",
            "email": "admin@ultratech.com", "user_id": "admin",
            "password": "admin123",
        })
        self.assertEqual(response.status_code, 302)
        self.assertIn("/dashboard", response.headers["Location"])

    def test_wrong_password_is_refused(self):
        self.client.get("/logout")
        response = self.client.post("/login", data={
            "csrf_token": self.csrf(), "role": "admin",
            "email": "admin@ultratech.com", "user_id": "admin",
            "password": "definitely-wrong",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("Authentication failed", response.get_data(as_text=True))

    def test_right_password_wrong_clearance_is_refused(self):
        """Admin credentials must not authenticate on the employee tab."""
        self.client.get("/logout")
        response = self.client.post("/login", data={
            "csrf_token": self.csrf(), "role": "employee",
            "email": "admin@ultratech.com", "user_id": "admin",
            "password": "admin123",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("Authentication failed", response.get_data(as_text=True))

    def test_employee_cannot_reach_the_admin_console(self):
        self.sign_in("employee")
        response = self.client.get("/dashboard/admin")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/dashboard", response.headers["Location"])

    def test_admin_is_dispatched_to_the_admin_console(self):
        self.sign_in("admin")
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/dashboard/admin", response.headers["Location"])

    def test_employee_is_dispatched_to_the_operations_console(self):
        self.sign_in("employee")
        response = self.client.get("/dashboard")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/dashboard/employee", response.headers["Location"])

    def test_logout_ends_the_session(self):
        self.sign_in("admin")
        self.client.get("/logout")
        response = self.client.get("/dashboard")
        self.assertIn("/login", response.headers["Location"])

    def test_login_rotates_the_csrf_token(self):
        self.client.get("/logout")
        before = self.csrf()
        self.client.post("/login", data={
            "csrf_token": before, "role": "admin",
            "email": "admin@ultratech.com", "user_id": "admin",
            "password": "admin123",
        })
        after = self.csrf("/predict")
        self.assertTrue(after)
        self.assertNotEqual(before, after,
                            "session fixation guard: token must rotate")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
class TestPages(ClientCase):

    def setUp(self) -> None:
        self.sign_in("admin")

    def test_every_page_renders(self):
        for path in PAGES:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, path)
                self.assertGreater(len(response.get_data()), 4000, path)

    def test_pages_carry_no_unrendered_placeholders(self):
        """Guards against template variables leaking as None or Undefined."""
        for path in PAGES:
            with self.subTest(path=path):
                body = self.client.get(path).get_data(as_text=True)
                self.assertNotIn("Undefined", body, path)
                self.assertNotIn("{{", body, path)
                self.assertNotIn("jinja2.exceptions", body, path)

    def test_login_page_is_reachable_without_a_session(self):
        self.client.get("/logout")
        response = self.client.get("/login")
        self.assertEqual(response.status_code, 200)
        self.assertIn("AI Maintenance Console", response.get_data(as_text=True))

    def test_shell_chrome_is_present(self):
        body = self.client.get("/dashboard/admin").get_data(as_text=True)
        self.assertIn('id="live-time"', body)
        self.assertIn('id="nav-toggle"', body)
        self.assertIn('id="chat-fab"', body)
        self.assertIn("favicon.svg", body)
        self.assertIn("scada.css", body)

    def test_every_sidebar_link_resolves(self):
        body = self.client.get("/dashboard/admin").get_data(as_text=True)
        for section in config.NAV_SECTIONS:
            for item in section["items"]:
                with self.subTest(label=item["label"]):
                    self.assertIn(item["label"], body)

    def test_unknown_page_returns_the_error_screen(self):
        response = self.client.get("/no-such-screen")
        self.assertEqual(response.status_code, 404)
        self.assertIn("Screen not found", response.get_data(as_text=True))

    def test_static_assets_are_served(self):
        for asset in ["/static/css/scada.css", "/static/js/app.js",
                      "/static/js/charts.js", "/static/img/favicon.svg"]:
            with self.subTest(asset=asset):
                response = self.client.get(asset)
                self.assertEqual(response.status_code, 200, asset)

    def test_charts_receive_data(self):
        body = self.client.get("/executive-dashboard").get_data(as_text=True)
        self.assertIn("Cockpit.line", body)
        self.assertIn('"labels"', body)

    def test_kiln_page_shows_the_headline_numbers(self):
        body = self.client.get("/kiln-analytics").get_data(as_text=True)
        for expected in ["332", "51", "659.7", "91.7", "156.2", "12.9"]:
            self.assertIn(expected, body, expected)

    def test_feature_intelligence_shows_the_collinearity_audit(self):
        body = self.client.get("/feature-intelligence").get_data(as_text=True)
        self.assertIn("Collinearity audit", body)
        self.assertIn("Variance inflation factor", body)
        self.assertIn("Thermal coupling", body)
        self.assertIn("Mechanical coupling", body)


# ---------------------------------------------------------------------------
# JSON API
# ---------------------------------------------------------------------------
class TestApi(ClientCase):

    def setUp(self) -> None:
        self.sign_in("admin")

    def test_all_get_endpoints_respond(self):
        for path in API_GET_ENDPOINTS:
            with self.subTest(path=path):
                response = self.client.get(path)
                self.assertEqual(response.status_code, 200, path)
                payload = response.get_json()
                self.assertIn("generated_at", payload, path)
                self.assertIn("platform", payload, path)

    def test_endpoint_count_matches_the_documented_eight(self):
        rules = [rule for rule in self.application.app.url_map.iter_rules()
                 if str(rule).startswith("/api/")]
        self.assertEqual(len(rules), 8, sorted(str(r) for r in rules))

    def test_machines_endpoint_lists_the_whole_fleet(self):
        payload = self.client.get("/api/machines").get_json()
        self.assertEqual(payload["count"], len(config.MACHINES))
        self.assertEqual(len(payload["machines"]), len(config.MACHINES))

    def test_fleet_status_matches_its_own_summary(self):
        payload = self.client.get("/api/fleet-status").get_json()
        self.assertEqual(len(payload["machines"]),
                         payload["summary"]["total"])
        for machine in payload["machines"]:
            self.assertAlmostEqual(
                machine["health_score"],
                100.0 - machine["failure_prob"], places=1)

    def test_kpis_expose_live_and_archive_blocks(self):
        payload = self.client.get("/api/kpis").get_json()
        self.assertIn("live", payload)
        self.assertIn("archive", payload)
        self.assertIn("trend", payload)

    def test_kiln_stats_carry_the_specified_figures(self):
        payload = self.client.get("/api/kiln-stats").get_json()
        self.assertEqual(payload["kpis"]["stoppages"], 51)
        self.assertEqual(payload["kpis"]["days"], 332)

    def test_predictions_endpoint_respects_its_limit(self):
        payload = self.client.get("/api/predictions?limit=2").get_json()
        self.assertLessEqual(len(payload["records"]), 2)

    def test_chat_endpoint_answers_from_live_data(self):
        token = self.token_after_login()
        response = self.client.post(
            "/api/chat", json={"message": "fleet status"},
            headers={"X-CSRF-Token": token})
        self.assertEqual(response.status_code, 200)
        self.assertIn("Fleet of", response.get_json()["reply"])

    def test_chat_recognises_a_machine_id(self):
        token = self.token_after_login()
        response = self.client.post(
            "/api/chat", json={"message": "how is KLN-01 doing?"},
            headers={"X-CSRF-Token": token})
        self.assertIn("KLN-01", response.get_json()["reply"])

    def test_chat_handles_nonsense_gracefully(self):
        token = self.token_after_login()
        response = self.client.post(
            "/api/chat", json={"message": "qwertyuiop"},
            headers={"X-CSRF-Token": token})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.get_json()["reply"])


# ---------------------------------------------------------------------------
# Prediction workflow
# ---------------------------------------------------------------------------
class TestPredictionWorkflow(ClientCase):

    def setUp(self) -> None:
        self.sign_in("admin")

    def test_prediction_is_stored_and_visible_downstream(self):
        token = self.token_after_login()
        before = self.client.get("/api/predictions").get_json()["stats"]["total"]

        response = self.client.post("/predict", data={
            "csrf_token": token, "machine_id": "FAN-07", "machine_type": "M",
            "air_temperature_k": "303.2", "process_temperature_k": "314.7",
            "rotational_speed_rpm": "1497", "torque_nm": "34.5",
            "tool_wear_min": "39", "runtime_hours": "6100",
        })
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Prediction result", body)
        self.assertIn("Health score", body)

        after = self.client.get("/api/predictions").get_json()
        self.assertEqual(after["stats"]["total"], before + 1)
        newest = after["records"][0]
        self.assertEqual(newest["machine_id"], "FAN-07")
        self.assertAlmostEqual(
            newest["health_score"], 100.0 - newest["failure_prob"], places=1)

        history = self.client.get("/history").get_data(as_text=True)
        self.assertIn("FAN-07", history)

    def test_invalid_reading_is_rejected_without_storing(self):
        token = self.token_after_login()
        before = self.client.get("/api/predictions").get_json()["stats"]["total"]

        response = self.client.post("/predict", data={
            "csrf_token": token, "machine_id": "", "machine_type": "M",
            "air_temperature_k": "300.0", "process_temperature_k": "310.0",
            "rotational_speed_rpm": "1500", "torque_nm": "9999",
            "tool_wear_min": "20", "runtime_hours": "100",
        })
        self.assertEqual(response.status_code, 200)
        after = self.client.get("/api/predictions").get_json()["stats"]["total"]
        self.assertEqual(after, before, "invalid input must not be persisted")

    def test_cockpit_preloads_a_live_machine_reading(self):
        response = self.client.get("/predict?machine_id=KLN-01")
        self.assertEqual(response.status_code, 200)
        self.assertIn("KLN-01", response.get_data(as_text=True))

    def test_cockpit_loads_a_preset(self):
        response = self.client.get("/predict?preset=critical")
        self.assertEqual(response.status_code, 200)
        self.assertIn("Critical Failure", response.get_data(as_text=True))

    def test_history_filters_by_status(self):
        response = self.client.get("/history?status=Excellent")
        self.assertEqual(response.status_code, 200)

    def test_history_search_runs(self):
        response = self.client.get("/history?q=FAN-07")
        self.assertEqual(response.status_code, 200)

    def test_csv_export_is_well_formed(self):
        token = self.token_after_login()
        self.client.post("/predict", data={
            "csrf_token": token, "machine_id": "KLN-02", "machine_type": "H",
            "air_temperature_k": "300.0", "process_temperature_k": "310.5",
            "rotational_speed_rpm": "1500", "torque_nm": "40.0",
            "tool_wear_min": "25", "runtime_hours": "2200",
        })
        response = self.client.get("/history/export.csv")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/csv", response.headers["Content-Type"])
        self.assertIn("attachment", response.headers["Content-Disposition"])

        rows = list(csv.DictReader(
            io.StringIO(response.get_data(as_text=True))))
        self.assertGreaterEqual(len(rows), 1)
        self.assertIn("health_score", rows[0])
        self.assertIn("machine_id", rows[0])


# ---------------------------------------------------------------------------
# Alerting
# ---------------------------------------------------------------------------
class TestAlerting(ClientCase):

    def setUp(self) -> None:
        self.sign_in("admin")

    def test_gateways_are_disabled_until_configured(self):
        from services import alert_service

        status = alert_service.gateway_status()
        self.assertIn(status["email"]["label"],
                      ("DISABLED", "NOT CONFIGURED"))
        self.assertIn(status["sms"]["label"],
                      ("DISABLED", "NOT CONFIGURED"))

    def test_fleet_scan_raises_alerts_then_deduplicates(self):
        from services import database

        token = self.token_after_login()
        self.client.post("/alerts/scan", data={"csrf_token": token})
        first = database.alert_counts()["total"]
        self.assertGreater(first, 0, "degraded machines should raise alerts")

        self.client.post("/alerts/scan", data={"csrf_token": token})
        second = database.alert_counts()["total"]
        self.assertEqual(first, second,
                         "a repeat scan must not duplicate open alerts")

    def test_acknowledging_clears_the_open_count(self):
        from services import database

        token = self.token_after_login()
        self.client.post("/alerts/scan", data={"csrf_token": token})
        self.client.post("/alerts/acknowledge", data={"csrf_token": token})
        self.assertEqual(database.alert_counts()["unacknowledged"], 0)

    def test_settings_persist(self):
        from services import database

        token = self.token_after_login()
        self.client.post("/alerts/settings", data={
            "csrf_token": token,
            "recipient_email": "maintenance.head@ultratech.com",
            "recipient_phone": "+91 90000 00000",
            "severity": "warning",
        })
        settings = database.get_alert_settings()
        self.assertEqual(settings["recipient_email"],
                         "maintenance.head@ultratech.com")
        self.assertEqual(settings["severity"], "warning")

    def test_invalid_settings_are_refused(self):
        from services import database

        token = self.token_after_login()
        self.client.post("/alerts/settings", data={
            "csrf_token": token, "recipient_email": "nonsense",
            "severity": "critical",
        }, follow_redirects=True)
        self.assertNotEqual(
            database.get_alert_settings()["recipient_email"], "nonsense")

    def test_test_notification_is_recorded_not_transmitted(self):
        from services import database

        token = self.token_after_login()
        response = self.client.post("/alerts/test",
                                    data={"csrf_token": token},
                                    follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        titles = [alert["title"] for alert in database.list_alerts(limit=10)]
        self.assertTrue(any("test notification" in title.lower()
                            for title in titles))

    def test_dispatch_is_suppressed_below_the_threshold(self):
        from services import alert_service

        outcomes = alert_service.dispatch(
            "info", "Low severity", "Should not be dispatched.")
        joined = " ".join(outcomes).lower()
        self.assertTrue("suppressed" in joined or "no gateway" in joined)


# ---------------------------------------------------------------------------
# Account administration
# ---------------------------------------------------------------------------
class TestUserAdministration(ClientCase):

    def setUp(self) -> None:
        self.sign_in("admin")

    def test_seeded_accounts_exist(self):
        from services import database

        ids = database.get_user_ids()
        for seed in config.SEED_USERS:
            self.assertIn(seed["user_id"], ids)

    def test_create_revoke_and_restore(self):
        from services import database

        token = self.token_after_login()

        self.client.post("/admin/users/create", data={
            "csrf_token": token, "user_id": "UT-9001",
            "name": "Kiln Operator", "email": "kiln.op@ultratech.com",
            "password": "operator123", "role": "employee",
            "department": "Operations",
        })
        self.assertIn("UT-9001", database.get_user_ids())

        self.client.post("/admin/users/UT-9001/revoke",
                         data={"csrf_token": token})
        revoked = [user for user in database.list_users()
                   if user["user_id"] == "UT-9001"][0]
        self.assertEqual(revoked["active"], 0)

        self.client.post("/admin/users/UT-9001/restore",
                         data={"csrf_token": token})
        restored = [user for user in database.list_users()
                    if user["user_id"] == "UT-9001"][0]
        self.assertEqual(restored["active"], 1)

    def test_revoked_account_cannot_sign_in(self):
        token = self.token_after_login()
        self.client.post("/admin/users/create", data={
            "csrf_token": token, "user_id": "UT-9002", "name": "Temp Staff",
            "email": "temp.staff@ultratech.com", "password": "temp123456",
            "role": "employee", "department": "Operations",
        })
        self.client.post("/admin/users/UT-9002/revoke",
                         data={"csrf_token": token})

        self.client.get("/logout")
        response = self.client.post("/login", data={
            "csrf_token": self.csrf(), "role": "employee",
            "email": "temp.staff@ultratech.com", "user_id": "UT-9002",
            "password": "temp123456",
        })
        self.assertEqual(response.status_code, 200)
        self.assertIn("Authentication failed", response.get_data(as_text=True))

    def test_duplicate_email_is_refused(self):
        from services import database

        token = self.token_after_login()
        before = len(database.get_user_ids())
        self.client.post("/admin/users/create", data={
            "csrf_token": token, "user_id": "UT-9003", "name": "Clash Test",
            "email": "admin@ultratech.com", "password": "clash12345",
            "role": "employee", "department": "Operations",
        })
        self.assertEqual(len(database.get_user_ids()), before)

    def test_admin_cannot_revoke_their_own_session(self):
        from services import database

        token = self.token_after_login()
        self.client.post("/admin/users/admin/revoke",
                         data={"csrf_token": token})
        admin = [user for user in database.list_users()
                 if user["user_id"] == "admin"][0]
        self.assertEqual(admin["active"], 1)

    def test_employee_cannot_create_accounts(self):
        from services import database

        self.sign_in("employee")
        token = self.csrf("/predict")
        before = len(database.get_user_ids())
        self.client.post("/admin/users/create", data={
            "csrf_token": token, "user_id": "UT-9004", "name": "Sneaky User",
            "email": "sneaky@ultratech.com", "password": "sneaky1234",
            "role": "admin", "department": "Management",
        })
        self.assertEqual(len(database.get_user_ids()), before)

    def test_actions_are_written_to_the_audit_trail(self):
        from services import database

        token = self.token_after_login()
        self.client.post("/admin/users/create", data={
            "csrf_token": token, "user_id": "UT-9005", "name": "Audit Check",
            "email": "audit.check@ultratech.com", "password": "audit12345",
            "role": "employee", "department": "Operations",
        })
        actions = [log["action"] for log in database.list_audit_logs(50)]
        self.assertIn("user_created", actions)
        self.assertIn("login", actions)


# ---------------------------------------------------------------------------
# Security
# ---------------------------------------------------------------------------
class TestSecurity(ClientCase):

    def setUp(self) -> None:
        self.sign_in("admin")

    def test_post_without_a_token_is_blocked(self):
        response = self.client.post("/alerts/scan", data={})
        self.assertEqual(response.status_code, 400)

    def test_post_with_a_forged_token_is_blocked(self):
        response = self.client.post(
            "/alerts/scan", data={"csrf_token": "forged-value"})
        self.assertEqual(response.status_code, 400)

    def test_api_post_without_a_token_header_is_blocked(self):
        response = self.client.post("/api/chat", json={"message": "hello"})
        self.assertEqual(response.status_code, 400)

    def test_session_cookie_is_hardened(self):
        self.assertTrue(
            self.application.app.config["SESSION_COOKIE_HTTPONLY"])
        self.assertEqual(
            self.application.app.config["SESSION_COOKIE_SAMESITE"], "Lax")

    def test_passwords_are_never_stored_in_clear_text(self):
        from services import database

        for user in database.list_users():
            self.assertNotIn("admin123", user["password_hash"])
            self.assertNotIn("employee123", user["password_hash"])
            self.assertTrue(len(user["password_hash"]) > 40)

    def test_secret_key_is_not_the_shipped_placeholder(self):
        """Fails until a real FLASK_SECRET_KEY is present in .env."""
        self.assertNotEqual(config.SECRET_KEY, "change-me-in-production",
                            "set FLASK_SECRET_KEY in .env")

    def test_no_live_credentials_are_committed(self):
        example = (PROJECT_ROOT / ".env.example").read_text(encoding="utf-8")
        for line in example.splitlines():
            if line.startswith(("PLATFORM_EMAIL_PASSWORD", "TWILIO_TOKEN",
                                "TWILIO_SID")):
                self.assertTrue(line.strip().endswith("="),
                                f"{line} must ship empty")

    def test_search_input_is_parameterised(self):
        """A quote in the search box must not break the SQL query."""
        response = self.client.get("/history?q=%27%20OR%201%3D1--")
        self.assertEqual(response.status_code, 200)


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
class TestFormatters(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        from utils import formatters

        cls.formatters = formatters

    def test_number_formatting(self):
        self.assertEqual(self.formatters.number(1234.567, 1), "1,234.6")
        self.assertEqual(self.formatters.number(1234.567, 0), "1,235")
        self.assertEqual(self.formatters.number(None), "-")

    def test_percent_and_hours(self):
        self.assertEqual(self.formatters.percent(91.75, 1), "91.8%")
        self.assertEqual(self.formatters.hours(1234, 0), "1,234 h")

    def test_status_colours_cover_every_band(self):
        for _, status, colour, _ in config.HEALTH_BANDS:
            self.assertEqual(self.formatters.status_colour(status), colour)

    def test_gauge_geometry(self):
        circumference = self.formatters.gauge_circumference(54)
        self.assertAlmostEqual(
            self.formatters.gauge_offset(100, 54), 0.0, places=1)
        self.assertAlmostEqual(
            self.formatters.gauge_offset(0, 54), circumference, places=1)

    def test_relative_time_handles_bad_input(self):
        self.assertEqual(self.formatters.relative_time(None), "-")
        self.assertEqual(self.formatters.relative_time("not-a-date"), "-")


if __name__ == "__main__":
    unittest.main(verbosity=2)
