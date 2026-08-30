# AI-Based Predictive Maintenance System

Predicting industrial machine failure before it happens.

A full-stack web platform that reads machine sensor values, predicts the
probability of failure using a Random Forest classifier, and converts that
prediction into an actionable maintenance decision: a health score, a remaining
useful life estimate, a recommended action, and a prioritised work order.

Built during industrial training at **Aditya Birla Group – UltraTech Cement**,
IT Department.

---

## Contents

- [Why this project exists](#why-this-project-exists)
- [What the system does](#what-the-system-does)
- [Technology stack](#technology-stack)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [The dataset](#the-dataset)
- [The machine learning model](#the-machine-learning-model)
- [From prediction to decision](#from-prediction-to-decision)
- [The monitored fleet](#the-monitored-fleet)
- [Database](#database)
- [Security](#security)
- [Alerting](#alerting)
- [Kiln stoppage analytics](#kiln-stoppage-analytics)
- [Getting started](#getting-started)
- [Testing](#testing)
- [Deployment](#deployment)
- [Configuration reference](#configuration-reference)
- [Known limitations](#known-limitations)
- [Future scope](#future-scope)

---

## Why this project exists

A cement plant runs continuously. Raw material is crushed, burnt in a rotary
kiln above 1,400 °C, then cooled, ground and packed. The machines operate as a
chain, so the failure of one critical unit — a kiln drive, a preheater fan, a
mill — halts the entire production line.

Maintenance practice has three levels of maturity:

| Approach | How it works | Problem |
|---|---|---|
| **Reactive** | Repair after the machine breaks | Most expensive. No warning, secondary damage, urgent spares |
| **Preventive** | Service on fixed dates or running hours | Wasteful. Healthy parts replaced; a fast-degrading part still fails in between |
| **Predictive** | Service when measured condition says it is needed | Requires the ability to interpret sensor data and forecast condition |

The third approach needs prediction, which is what machine learning provides.
This project is a working demonstration of it.

---

## What the system does

**Input:** five sensor readings plus the machine quality class.
**Output:** a decision an engineer can act on.

Thirteen role-protected screens:

| Screen | Purpose |
|---|---|
| Login portal | Role-based authentication, admin and employee clearance |
| Admin control centre | Create, revoke and restore employee accounts; audit trail |
| Operations dashboard | Live condition of every monitored asset |
| Predictive cockpit | Sensor entry by slider or keyboard, with operating-range guidance |
| Prediction result | Health gauge, failure probability, RUL, risk drivers, action plan |
| Prediction history | Searchable, filterable archive with CSV export |
| Alert centre | Critical and warning logs, fleet scan, notification gateways |
| Executive dashboard | Fleet health, failure risk, availability, estimated MTTR |
| Analytics dashboard | Condition distribution, health index, prediction volume trends |
| AI feature intelligence | Model card, metrics, feature importance, collinearity audit |
| Kiln stoppage analytics | Downtime, availability, MTBF, MTTR, cause Pareto |
| Maintenance advisor | Prioritised P1–P5 work-order schedule, departmental loading |
| Architecture / Roadmap | Design documentation screens |

Eight REST endpoints expose the same data as JSON:

| Method | Endpoint | Returns |
|---|---|---|
| GET | `/api/machines` | Monitored asset register |
| GET | `/api/fleet-status` | Live per-machine assessment |
| GET | `/api/predictions` | Stored prediction archive |
| GET | `/api/alerts` | Alerts and counters |
| GET | `/api/kpis` | Dashboard KPIs, trend, distribution |
| GET | `/api/kiln-stats` | Kiln availability analytics |
| GET | `/api/feature-importance` | Model card, metrics, collinearity audit |
| POST | `/api/chat` | Plant assistant reply |

All endpoints require an authenticated session.

---

## Technology stack

| Component | Technology | Why |
|---|---|---|
| Language | Python 3.13 | One language across data, model and web layers |
| Web framework | Flask 3.1 | Lightweight and explicit; factory pattern keeps routing thin |
| Machine learning | scikit-learn 1.8 | Mature Random Forest with a pipeline abstraction |
| Data handling | pandas 2.3, NumPy 2.4 | Standard tabular generation and analysis |
| Database | SQLite 3 | Zero-configuration embedded store |
| Model persistence | joblib 1.5 | Efficient serialisation of the fitted pipeline |
| Front end | Bootstrap 5.3, Chart.js 4.4 | Responsive layout and charts without a JS build step |
| Templating | Jinja2 | Server-rendered pages |
| Production server | gunicorn 23 | WSGI server for deployment |
| Testing | unittest (stdlib) | Runs on any Python, no extra dependency |

---

## Architecture

Three layers, each talking only to the one beneath it.

```
┌──────────────────────────────────────────────────────────────┐
│  PRESENTATION                                                │
│  13 Jinja2 pages · Bootstrap 5 · Chart.js 4 · dark SCADA CSS │
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  APPLICATION                                                 │
│  Flask factory · 5 blueprints · 7 services · 3 utils         │
│  auth · validation · prediction · alerting · 8 REST endpoints│
└──────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────┐
│  DATA & INTELLIGENCE                                         │
│  SQLite (5 tables) · Random Forest pipeline (joblib)         │
└──────────────────────────────────────────────────────────────┘
```

**Request flow for a prediction**

```
Browser
  → Flask blueprint (routes/predict.py)
  → input validator (utils/validators.py)
  → ColumnTransformer: StandardScaler + OneHotEncoder
  → RandomForestClassifier → failure probability
  → health score · RUL · priority (services/prediction_service.py)
  → SQLite persistence (services/database.py)
  → Jinja2 template + Chart.js render
```

The simulated sensor feed sits behind a clean service boundary, so replacing it
with real MQTT or OPC-UA tags requires no change to the model, the decision
layer or the interface.

---

## Project structure

```
1_PROJECT_Website/
├── app.py                  Application factory, CSRF guard, error handlers
├── wsgi.py                 Production entrypoint (runs bootstrap on import)
├── config.py               Every setting in one file
├── requirements.txt
├── render.yaml             Render deployment blueprint
├── Procfile                Start command for Railway / Koyeb / Heroku
├── .python-version         Pins Python 3.13.1
├── .env.example            Environment template (real .env is git-ignored)
│
├── scripts/
│   ├── generate_synthetic_dataset.py   Builds the 1,500-row telemetry file
│   ├── generate_kiln_dataset.py        Builds the 332-day stoppage history
│   └── train_model.py                  Trains, evaluates, saves the model
│
├── routes/                 Blueprints: auth, main, predict, history, api
├── services/               database, ml, prediction, fleet, alert, kiln, chatbot
├── utils/                  validators, formatters, recommendations
├── templates/              base + 13 pages + error + 4 partials
├── static/                 scada.css, app.js, charts.js, favicon.svg
├── tests/test_platform.py  115 automated tests
├── dataset/                Generated CSVs + measured data profile
├── models/                 Trained pipeline + metadata (built, not committed)
└── legacy/                 Early exploration scripts, kept for reference
```

---

## The dataset

No labelled run-to-failure history was available for a student project, so the
training data is generated. The schema follows the **AI4I 2020 predictive
maintenance** reference specification.

### The six inputs

| Input | Unit | Allowed range | Normal band | Meaning |
|---|---|---|---|---|
| Air temperature | K | 290 – 315 | 295 – 305 | Air around the machine |
| Process temperature | K | 300 – 332 | 305 – 315 | The machine while running |
| Rotational speed | RPM | 1000 – 3000 | 1300 – 1700 | Shaft speed |
| Torque | Nm | 3 – 80 | 30 – 50 | Load on the drive train |
| Tool wear | min | 0 – 260 | 0 – 200 | Minutes the wear parts have run |
| Machine class | L / M / H | — | — | Light, Medium, Heavy duty |

1,500 rows, random seed 42, machine mix L 60% / M 30% / H 10%.

### Collinearity is preserved on purpose

This is the most important design decision in the project.

A naive generator draws every column independently. That destroys the physical
structure of real telemetry and produces a model whose feature importances are
meaningless. In a real machine the sensors are **not** independent — they are
different views of the same physics. Three real dependencies were therefore
built in and then measured to confirm they survived:

| Coupling | How it was generated | Expected | **Measured** |
|---|---|---|---|
| **Thermal** | `process_temp = air_temp + N(10, 1)` — ambient heat enters the machine | strong positive | **r = +0.837** |
| **Mechanical** | `speed = 60·P / (2π·torque)` — speed derived from power, not drawn | strong negative | **r = −0.777** |
| **Duty** | Wear rate ×1.18 for L, ×1.00 for M, ×0.84 for H | light duty wears faster | **r = +0.165** |

Air temperature itself is a mean-reverting random walk
(`v = 0.94·v_prev + N(0, 0.75)`), not white noise, because real ambient
conditions drift across a shift.

**Variance Inflation Factor** — how much of each column is explained by the
others. 1.0 means fully independent:

| Channel | VIF | Reading |
|---|---|---|
| Air temperature | 3.34 | shares information with process temp |
| Process temperature | 3.34 | shares information with air temp |
| Rotational speed | 2.54 | shares information with torque |
| Torque | 2.53 | shares information with speed |
| Tool wear | 1.01 | independent, as a cumulative counter should be |

Computed with NumPy only, via `VIF = 1 / (1 − R²)` from `np.linalg.lstsq`.

### Labels come from physics, not a coin flip

Six physical failure modes, each a **graded hazard** rather than a hard
threshold — risk ramps up through a transition band using a logistic curve:

| Mode | Physical condition | Threshold | Ramp |
|---|---|---|---|
| TWF | Tool wear exhausted | 208 min | 8 min |
| HDF | Air-to-process gap collapsed **and** airflow low | 8.8 K / 1410 RPM | 0.34 K / 45 RPM |
| PWF | Delivered power outside envelope | < 3700 W or > 9100 W | 150 W |
| OSF | Overstrain: wear × torque past class limit | L 9500, M 10500, H 11500 | 480 |
| OSP | Bearing overspeed | 2545 RPM | 48 RPM |
| TOL | Torque overload | 56.2 Nm | 3.0 Nm |
| RNF | Random, unexplained | 0.4% flat | — |

No single mode may exceed 0.90, so nothing is a guaranteed failure. Modes
combine as competing independent risks:

```
total_hazard = 1 − Π(1 − p_i)
failure      = random() < total_hazard
```

### Why graded hazards, not hard cut-offs

The first version used hard thresholds. The model scored **94% accuracy** but
returned only ~0% or ~100% — nothing in between. That made the Good and Warning
health bands mathematically unreachable, so no machine could ever be reported as
*slowly degrading*, which is the whole point of predictive maintenance.

Replacing the cut-offs with graded ramps dropped accuracy to 84% and made the
system genuinely useful. **Accepting a lower score for a better system was the
key engineering decision of this project.**

### Calibration check

Final dataset: 1,500 rows, **33.93% failures**.

| Hazard band | Rows | Actually failed |
|---|---|---|
| 0.00 – 0.10 | 798 | 2.3% |
| 0.10 – 0.25 | 113 | 15.0% |
| 0.25 – 0.45 | 93 | 39.8% |
| 0.45 – 0.65 | 71 | 56.3% |
| 0.65 – 0.85 | 78 | 78.2% |
| 0.85 – 1.00 | 347 | 91.4% |

Predicted risk matches observed failure rate at every level. That is what
*calibrated* means.

---

## The machine learning model

### Algorithm

**RandomForestClassifier** — an ensemble of 150 decision trees, each grown on a
bootstrap sample and each considering a random subset of features at every
split. Their predictions are averaged.

Chosen for four concrete reasons:

1. Handles non-linear interactions between sensors without feature engineering
2. Robust to the correlated inputs this dataset deliberately contains
3. Returns a **probability**, which the decision layer requires
4. Reports **feature importance**, making it explainable to an engineer

### Pipeline

Preprocessing and model are bound into one object, so the transformations applied
at prediction time are guaranteed to be those fitted during training:

```python
Pipeline([
    ("preprocessor", ColumnTransformer([
        ("num", StandardScaler(),  [5 numeric sensors]),
        ("cat", OneHotEncoder(handle_unknown="ignore"), ["machine_type"]),
    ])),
    ("classifier", RandomForestClassifier(n_estimators=150, max_depth=16,
                                          min_samples_leaf=1, random_state=42)),
])
```

| Setting | Value |
|---|---|
| Trees | 150 |
| Max depth | 16 |
| Min samples per leaf | 1 |
| Split | 1,125 train / 375 test, stratified |
| Random state | 42 |
| Validation | 5-fold stratified cross-validation |

### Results on held-out data

| Metric | Value | Meaning |
|---|---|---|
| Accuracy | **84.00%** | Overall correct predictions |
| Precision | **80.18%** | Of flagged failures, how many were real |
| Recall | **70.08%** | Of real failures, how many were caught |
| F1 score | **74.79%** | Balance of the two |
| ROC-AUC | **87.68%** | Ranks a failing machine above a healthy one |
| CV F1 | **76.12% ± 2.50%** | Stable across five different splits |

Confusion matrix on the 375 unseen readings:

| | Predicted healthy | Predicted failure |
|---|---|---|
| **Actually healthy** | 226 | 22 (false alarm) |
| **Actually failure** | 38 (missed) | 89 |

On the precision/recall gap: industrially, **recall matters more** — a missed
failure stops production, a false alarm costs an inspection. The platform
effectively favours recall already, because it acts on the continuous
probability and the health bands rather than a fixed 50% cut-off.

### Feature importance

| Rank | Feature | Importance | Why it ranks there |
|---|---|---|---|
| 1 | Torque | **31.88%** | Direct mechanical load; drives overload and overstrain |
| 2 | Rotational speed | **25.66%** | With torque determines power; drives overspeed |
| 3 | Tool wear | **18.02%** | Dominant slow-degradation channel |
| 4 | Process temperature | **11.73%** | Part of the heat-dissipation gap |
| 5 | Air temperature | **10.27%** | Ambient baseline for cooling margin |
| 6 | Machine class | **2.43%** | Shifts the overstrain limit |

Torque and speed together account for **57.5%** of the decision. Their product
is mechanical power, and power excursions are the most common failure mechanism
in the data. **The model learned real physics, not noise** — which is the
strongest available evidence that the dataset is sound.

---

## From prediction to decision

A probability alone is useless to an engineer. Four conversions turn it into
something actionable.

### Health score

```
health_score = 100 − failure_probability
```

Deliberately trivial, so it can be explained in one sentence and audited against
the raw model output.

### Risk bands

| Band | Health | Recommended action | Next service |
|---|---|---|---|
| Excellent | 85 – 100 | Monitor | 90 days |
| Good | 65 – 85 | Schedule inspection | 60 days |
| Warning | 45 – 65 | Plan maintenance | 30 days |
| Critical | below 45 | Immediate shutdown | 7 days |

### Remaining Useful Life

**RUL is a documented heuristic, not a model output.** The classifier answers
*whether* a machine will fail, not *when*. Proper time-to-failure prediction
needs run-to-failure histories or survival analysis, neither of which was
available. Being explicit about this was a deliberate choice — presenting a
heuristic as an ML prediction would misrepresent the system.

```
RUL = 7600 h × wear_term × health_term × stress_term
```

| Term | Formula | Purpose |
|---|---|---|
| wear_term | `1 − 0.55 × (tool_wear / 253)` | Life consumed by elapsed wear |
| health_term | `0.45 + 0.55 × (1 − failure_prob)` | Scale by model confidence, with a floor |
| stress_term | `max(1 − penalty, 0.35)` | Penalty for operating off the nominal duty point |

```
penalty     = 0.20·thermal_dev + 0.15·speed_dev + 0.15·torque_dev
thermal_dev = worst of (gap−10)/10  or  (10−gap)/10 × 1.5   ← both extremes hurt
speed_dev   = max(0, (rpm − 1700) / 1500)
torque_dev  = max(0, (torque − 50) / 40)
```

All three terms are displayed alongside the result, so the estimate can be
inspected rather than trusted blindly.

### Work-order priority

| Code | Level | Assigned when | Response | Man-hours |
|---|---|---|---|---|
| P1 | Emergency | Critical band and prob ≥ 80%, or ≥ 2 hard limits breached | Immediate | 6.0 |
| P2 | Urgent | Critical band, or Warning with a hard limit breached | Within 24 h | 4.0 |
| P3 | High | Warning band, no hard limit breached | Within 7 days | 2.5 |
| P4 | Scheduled | Good band with risk ≥ 30% | Next shutdown | 1.5 |
| P5 | Preventive | Excellent or Good, nothing flagged | Routine round | 0.5 |

### Risk drivers

Every prediction names the sensors pushing risk up, worst first, with a severity
and a plain-language explanation, plus corrective and preventive steps.

| Driver | Critical when |
|---|---|
| Heat dissipation | gap < 8.6 K |
| Bearing overspeed | ≥ 2570 RPM |
| Torque overload | ≥ 56.5 Nm |
| Tool wear | ≥ 200 min (warning at 150) |
| Overstrain | wear × torque > class limit |
| Power draw | outside 3600 – 9200 W |

Driver limits sit slightly *outside* the hazard midpoints used in generation
(2570 vs 2545, 56.5 vs 56.2), so a machine merely approaching a limit is reported
as elevated risk rather than an outright breach.

---

## The monitored fleet

| ID | Machine | Class | Department | Duty profile |
|---|---|---|---|---|
| KLN-01 | Rotary Kiln Line 1 | H | Mechanical | warning |
| KLN-02 | Rotary Kiln Line 2 | H | Mechanical | excellent |
| BLR-03 | Bag Filter Blower Unit | M | Electrical | critical |
| RMM-04 | Raw Material Ball Mill | H | Mechanical | good |
| CML-05 | Coal Mill Drive | M | Mechanical | excellent |
| CLC-06 | Clinker Cooler Fan | M | Instrumentation | good |
| FAN-07 | Preheater Fan | M | Electrical | excellent |
| CRS-08 | Limestone Crusher | L | Mechanical | critical |
| PKR-09 | Cement Packing Line | L | Instrumentation | warning |

Consistently yields **5 healthy / 2 warning / 2 critical** and **P1 0 · P2 2 ·
P3 2 · P4 0 · P5 5**, verified stable across 40 consecutive refresh windows.

**How the simulator works.** There is no real sensor link, and the system says so
openly. Time is cut into 30-second buckets; the seed is
`CRC32(machine_id + bucket + attempt)`, so every page rendered in the same bucket
shows identical numbers and no two dashboards can disagree. For each machine, 14
candidate operating points are drawn from its physical envelope, all scored in
one batched model call, and the candidate exhibiting the intended condition is
kept. Envelopes respect the power constraint (torque × speed inside 3.7–9.1 kW).

The simulator only chooses *which operating point a machine sits at*, exactly as
reality would. **The model still performs every classification.**

---

## Database

SQLite at `instance/predictive_maintenance.db`.

| Table | Contents |
|---|---|
| `users` | user_id, name, email, password_hash, role, department, active, last_login |
| `predictions` | All six inputs plus failure_prob, health_score, status, rul_hours, next_service_days, action, priority, created_by |
| `alerts` | machine_id, severity, title, message, channel, acknowledged, timestamps |
| `alert_settings` | Single row: email/SMS toggles, recipients, severity threshold |
| `audit_logs` | actor, action, detail, timestamp |

Indexes on `predictions(created_at DESC)`, `predictions(machine_id)`,
`alerts(created_at DESC)`. Connections use a context manager: commit on success,
**rollback on exception**, always closed.

---

## Security

| Measure | Implementation |
|---|---|
| Authentication | Every page and endpoint behind a session login |
| Password storage | Werkzeug PBKDF2 hashes, never clear text |
| Role authorisation | `@login_required` / `@admin_required` decorators |
| CSRF protection | Per-session token on every state-changing request, `secrets.compare_digest` |
| Session fixation | Token rotated on login |
| SQL injection | Every statement parameterised |
| Input validation | Sensors range-checked before reaching the model |
| Cookie hardening | HttpOnly, SameSite=Lax, Secure in production, 4-hour lifetime |
| Reverse proxy | ProxyFix in production so redirects keep the https scheme |
| Secret management | `.env` git-ignored; repo ships only an empty template |
| Startup guard | Production refuses to boot on placeholder secret or demo passwords |
| Seed re-sync | Seeded passwords re-synced from environment on every start |
| Auditability | Login, logout, failed login, prediction, account and config changes logged |

**Note on seed re-sync:** an insert-only seed would silently leave the previous
password working after an operator changed the environment variable. Seed
accounts therefore have their hash re-synced on every start. Accounts created
through the admin console are untouched.

---

## Alerting

- Scans the fleet and raises warning or critical alerts, with the reason drawn
  from the risk drivers
- **De-duplicated within 15 minutes**, so a fault still present does not
  re-alert on every page render
- Acknowledgement singly, by severity, or all, with timestamps
- Email over SMTP and SMS over Twilio, both behind config flags, **disabled by
  default**
- **Fails soft:** with no credentials the alert is recorded internally and the
  app continues. No page render ever blocks on a network call

---

## Kiln stoppage analytics

A dedicated module for the most critical asset, over a synthetic 332-day history.

| Measure | Result |
|---|---|
| Window | 332 days (04-05-2025 → 31-03-2026) |
| Total stoppages | 51 |
| Total downtime | 659.7 hours |
| Availability | 91.7% |
| MTBF | 156.2 h (calendar basis) / 143.3 h (uptime basis) |
| MTTR | 12.9 h |
| Production loss | ~91,116 tonnes clinker |

Both MTBF conventions are published side by side so the definition is never
ambiguous. Downtime Pareto: refractory and coating failure accounts for **51.1%**
of all lost hours across only 10 events — which tells management exactly where
corrective investment pays off.

Stoppages never overlap; a minimum 20-hour gap is enforced and durations are
redistributed iteratively to land on the exact target total.

---

## Getting started

### Requirements

- Python 3.13
- Windows, macOS or Linux

### Install and run

```bash
git clone https://github.com/Vedant251205/-Ai-predictive-maintenance.git
cd -Ai-predictive-maintenance

python -m pip install -r requirements.txt
python app.py
```

Open **http://127.0.0.1:5000**

On first run the app generates both datasets, trains the model and creates the
database automatically. No manual preparation step.

| Role | Email | ID | Password |
|---|---|---|---|
| Administrator | admin@ultratech.com | admin | admin123 |
| Employee | employee@ultratech.com | employee | employee123 |

Local defaults only. They are overridden by environment variables in production,
and the login page has a button that fills them in for you during development.

### Regenerating artefacts

```bash
python scripts/generate_synthetic_dataset.py   # data + collinearity audit
python scripts/generate_kiln_dataset.py        # kiln history
python scripts/train_model.py                  # retrain and re-evaluate
```

Seeds are fixed (42 for telemetry, 7 for kiln), so output is reproducible.

---

## Testing

```bash
python -m unittest discover -s tests          # quiet
python -m unittest discover -s tests -v       # verbose
```

**115 tests, all passing**, in roughly 15 seconds. Built on the standard library
so no extra packages are needed, and run against a temporary database so real
data is never touched.

| Area | Sample assertions |
|---|---|
| Dataset | Correlations hold, VIF in range, hazards calibrated, `power_w` matches torque × angular velocity |
| Model | 150 trees, correct preprocessing, importances sum to 100%, overloaded scores worse than healthy |
| Decision layer | Health is exactly `100 − prob`, RUL falls as wear and risk rise, service within band ceiling |
| Fleet | Deterministic per window, 5/2/2 mix stable, power inside envelope |
| Kiln | Figures match spec, stoppages never overlap, counts reconcile |
| Web | All 13 pages render with no template errors, all 8 endpoints respond |
| Security | Token-less and forged POSTs rejected, anonymous redirected, employee blocked from admin, no clear-text passwords, injection string safe |

The suite caught two real defects during development: a template expression that
resolved to a dictionary method instead of its key, and a test that assumed an
anonymous session while an earlier test had left one signed in.

---

## Deployment

Configured for **Render** via `render.yaml`; `Procfile` covers Railway, Koyeb and
Heroku-style hosts.

1. Push this repository to GitHub
2. On Render: **New → Blueprint** → select the repository
3. Set `ADMIN_PASSWORD` and `EMPLOYEE_PASSWORD` when prompted
4. Wait about five minutes for the first build

The build step installs dependencies, generates both datasets and trains the
model on the server, so no large binaries are committed. Generation is seeded, so
the deployed model is identical to the one verified locally.

`wsgi.py` exists because a WSGI server *imports* the module rather than executing
it, so `bootstrap()` would never run and the first request would arrive with no
database and no model.

**Free-tier behaviour:** the service sleeps after 15 minutes idle (~50 s to wake)
and has no persistent disk, so stored predictions reset on restart. The site
always returns fully working because `bootstrap()` rebuilds the schema and
re-seeds accounts — it simply forgets prior predictions.

---

## Configuration reference

Everything lives in `config.py`: branding, the asset register, health bands, RUL
constants, priorities, alert thresholds, operating ranges, navigation and paths.
Change one file and the whole platform changes.

Environment variables (set in `.env` locally, in the host dashboard in production):

| Variable | Default | Purpose |
|---|---|---|
| `APP_ENV` | `development` | Set to `production` to harden cookies and hide demo helpers |
| `FLASK_SECRET_KEY` | placeholder | Session signing key. Production refuses to start on the default |
| `PORT` | `5000` | Supplied automatically by the host |
| `DATABASE_PATH` | `instance/…db` | Point at a writable volume if the project dir is read-only |
| `ADMIN_PASSWORD` | `admin123` | Seeded admin password |
| `EMPLOYEE_PASSWORD` | `employee123` | Seeded employee password |
| `SHOW_DEMO_KEYS` | `1` dev, `0` prod | Shows the credential autofill button |
| `EMAIL_ENABLED` | `0` | Enable SMTP dispatch |
| `SMS_ENABLED` | `0` | Enable Twilio dispatch |

---

## Known limitations

Stated plainly rather than hidden.

1. **Synthetic data.** Engineered to reproduce the statistical and physical
   structure of real telemetry, but not an actual plant's failure history.
2. **No live sensors.** A simulator stands in at a defined service boundary.
3. **RUL is a heuristic**, not a model prediction.
4. **Flask development server** locally; production needs gunicorn behind TLS.
5. **API is session-authenticated** — a real SCADA integration needs tokens or mTLS.
6. **SQLite** suits a single process; multiple writers need a networked database.

---

## Future scope

- Replace the simulator with MQTT or OPC-UA subscriptions to real plant tags
- Retrain on the plant's own run-to-failure records
- Replace the RUL heuristic with survival analysis or a gradient-boosted regressor
- Write recommended work orders directly into the plant CMMS
- Containerise, move to a networked database, serve behind TLS with token auth
- Extend to vibration and acoustic channels, the most informative signals for
  bearing and gearbox faults

---

## Acknowledgement

Developed during industrial training at Aditya Birla Group – UltraTech Cement,
IT Department, under the supervision of the reporting manager, and submitted to
the School of Computer Engineering, KIIT Deemed to be University.
