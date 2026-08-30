"""
Plant assistant.

A deterministic, intent-matching assistant. It answers from live platform state
(fleet snapshot, model metadata, kiln analytics, stored alerts) rather than
generating text, so every figure it quotes is one a user can verify elsewhere in
the interface.
"""

from __future__ import annotations

import re

import config
from services import database, fleet_service, kiln_service, ml_service

SUGGESTIONS = [
    "Fleet status",
    "Which machines are critical?",
    "How accurate is the model?",
    "What drives failures?",
    "Kiln downtime",
    "Explain RUL",
]


def _fleet_status() -> str:
    fleet = fleet_service.snapshot()
    summary = fleet_service.summary(fleet)
    lines = [
        f"Fleet of {summary['total']} machines: {summary['healthy']} healthy, "
        f"{summary['warning']} warning, {summary['critical']} critical.",
        f"Average health {summary['avg_health']}%, average remaining useful "
        f"life {summary['avg_rul_hours']:,.0f} h "
        f"({summary['avg_rul_days']} days).",
    ]
    counts = summary["priority_counts"]
    lines.append(
        "Open work orders: "
        + ", ".join(f"{code} {counts[code]}" for code in ("P1", "P2", "P3", "P4", "P5"))
        + "."
    )
    return " ".join(lines)


def _critical() -> str:
    fleet = fleet_service.snapshot()
    flagged = [item for item in fleet if item["status"] in ("Critical", "Warning")]
    if not flagged:
        return "Nothing is outside its healthy band right now. Every machine is Good or Excellent."

    flagged.sort(key=lambda item: item["health_score"])
    parts = [
        f"{item['machine_id']} ({item['machine_name']}) - {item['status']}, "
        f"health {item['health_score']}%, failure probability "
        f"{item['failure_prob']}%, priority {item['priority']}"
        for item in flagged
    ]
    return "Machines needing attention: " + "; ".join(parts) + "."


def _machine(machine_id: str) -> str:
    item = fleet_service.machine(machine_id)
    if item is None:
        return f"I have no live reading for {machine_id}."
    drivers = item["drivers"]
    driver_text = (
        " Main factors: " + "; ".join(entry["message"] for entry in drivers[:2])
        if drivers
        else " No parameter is outside its operating envelope."
    )
    return (
        f"{item['machine_id']} - {item['machine_name']} ({item['department']}). "
        f"Status {item['status']}, health {item['health_score']}%, failure "
        f"probability {item['failure_prob']}%, remaining useful life "
        f"{item['rul_hours']:,} h, next service in "
        f"{item['next_service_days']} days, priority {item['priority']} "
        f"({item['priority_label']}). Recommended action: {item['action']}."
        + driver_text
    )


def _model() -> str:
    card = ml_service.model_card()
    metrics = ml_service.metrics()
    return (
        f"The classifier is a {card['algorithm']} with {card['n_estimators']} "
        f"trees behind a {card['preprocessor']}. Trained on "
        f"{card['train_rows']} of {card['dataset_rows']} synthetic readings "
        f"({card['positive_rate_pct']}% failures). Held-out test performance: "
        f"accuracy {metrics.get('accuracy')}%, precision "
        f"{metrics.get('precision')}%, recall {metrics.get('recall')}%, "
        f"F1 {metrics.get('f1')}%, ROC-AUC {metrics.get('roc_auc')}%. "
        f"5-fold cross-validated F1 is {metrics.get('cv_f1_mean')}% "
        f"+/- {metrics.get('cv_f1_std')}%."
    )


def _importance() -> str:
    ranked = ml_service.feature_importance()
    if not ranked:
        return "Feature importances are not available until the model is trained."
    parts = [f"{entry['label']} {entry['pct']}%" for entry in ranked]
    return (
        "Failure risk is driven mostly by mechanical load. Ranked importance: "
        + ", ".join(parts)
        + ". Torque and rotational speed dominate because they jointly "
          "determine delivered power, which is the most common failure mode."
    )


def _collinearity() -> str:
    audit = ml_service.collinearity()
    relationships = audit.get("preserved_relationships", [])
    if not relationships:
        return "The collinearity audit is not available yet."
    parts = [
        f"{entry['name']} ({entry['pair']}) r = {entry['observed_r']:+.3f}"
        for entry in relationships
    ]
    vif = audit.get("vif", {})
    vif_text = ", ".join(
        f"{config.FEATURE_LABELS.get(key, key)} {value}" for key, value in vif.items()
    )
    return (
        "The synthetic dataset preserves the physical relationships between "
        "sensors rather than drawing each column independently: "
        + "; ".join(parts)
        + f". Variance inflation factors: {vif_text}."
    )


def _rul() -> str:
    cfg = config.RUL
    return (
        "Remaining Useful Life is a documented heuristic, not a model output - "
        "the classifier answers whether a unit will fail, not when. RUL = "
        f"{cfg['base_life_hours']:,.0f} h baseline x a wear term x a health "
        "term x a stress term. The wear term consumes life in proportion to "
        "elapsed tool wear, the health term scales by the model's confidence, "
        "and the stress term penalises operation away from the nominal duty "
        "point on temperature, speed and torque."
    )


def _kiln() -> str:
    try:
        data = kiln_service.kpis()
    except kiln_service.KilnDataUnavailable:
        return "The kiln stoppage dataset has not been generated yet."
    causes = kiln_service.cause_breakdown()
    top = causes[0] if causes else None
    text = (
        f"Over {data['days']} days ({data['start_date']} to {data['end_date']}) "
        f"kiln {data['unit']} recorded {data['stoppages']} stoppages totalling "
        f"{data['downtime_hours']:,} h of downtime. Availability "
        f"{data['availability']}%, MTBF {data['mtbf']} h, MTTR "
        f"{data['mttr']} h. Estimated production loss "
        f"{data['production_loss']:,.0f} tonnes."
    )
    if top:
        text += (
            f" The largest contributor is {top['cause']}: {top['events']} "
            f"events, {top['hours']} h, {top['share']}% of all downtime."
        )
    return text


def _alerts() -> str:
    counts = database.alert_counts()
    if counts["total"] == 0:
        return (
            "No alerts are stored yet. Open the Alert Center and run a fleet "
            "scan to raise alerts for any machine outside its healthy band."
        )
    return (
        f"{counts['total']} alerts stored: {counts['critical']} critical, "
        f"{counts['warning']} warning, {counts['unacknowledged']} still "
        "unacknowledged."
    )


def _thresholds() -> str:
    return (
        "Health score is the complement of the model's failure probability. "
        "Bands: Excellent 85-100, Good 65-85, Warning 45-65, Critical below "
        "45. Work orders map from those bands onto P1 Emergency through "
        "P5 Preventive."
    )


def _help() -> str:
    return (
        "I can report live fleet status, explain any machine by its id "
        "(for example KLN-01), summarise model performance and feature "
        "importance, explain how health score and RUL are calculated, and "
        "report kiln stoppage analytics or stored alerts."
    )


INTENTS = [
    (r"\b(hi|hello|hey|good (morning|afternoon|evening))\b",
     lambda: "Plant assistant online. Ask me about fleet status, a specific "
             "machine, the model, or kiln downtime."),
    (r"\b(help|what can you|commands|options)\b", _help),
    (r"\b(fleet|overall|overview|status|summary|how many machines)\b", _fleet_status),
    (r"\b(critical|urgent|danger|failing|worst|attention|risk machines)\b", _critical),
    (r"\b(accuracy|accurate|precision|recall|f1|auc|performance|model|algorithm|random forest)\b",
     _model),
    (r"\b(importance|drives|driver|feature|which sensor|most important)\b", _importance),
    (r"\b(collinear|collinearity|correlation|vif|dataset|synthetic|data quality)\b",
     _collinearity),
    (r"\b(rul|remaining useful life|how long|lifetime|life left)\b", _rul),
    (r"\b(kiln|stoppage|downtime|mtbf|mttr|availability)\b", _kiln),
    (r"\b(alert|notification|email|sms|warning list)\b", _alerts),
    (r"\b(health score|band|threshold|priority|p1|p2|p3|p4|p5)\b", _thresholds),
]


def reply(message: str) -> dict:
    text = (message or "").strip()
    if not text:
        return {
            "reply": "Ask me anything about the plant, the fleet or the model.",
            "suggestions": SUGGESTIONS,
        }

    lowered = text.lower()

    # A machine id anywhere in the question always wins: it is the most
    # specific thing the user could have asked about.
    match = re.search(r"\b([a-z]{3}-\d{2})\b", lowered)
    if match:
        machine_id = match.group(1).upper()
        if machine_id in config.MACHINE_INDEX:
            return {"reply": _machine(machine_id), "suggestions": SUGGESTIONS}

    for pattern, handler in INTENTS:
        if re.search(pattern, lowered):
            return {"reply": handler(), "suggestions": SUGGESTIONS}

    return {
        "reply": (
            "I did not follow that. I can help with fleet status, a machine id "
            "such as KLN-01, model performance, feature importance, RUL, kiln "
            "downtime or alerts."
        ),
        "suggestions": SUGGESTIONS,
    }
