"""
Explainability Layer
=======================

Produces per-alert SHAP feature attributions and converts them into a
human-readable rationale string for SOC analysts, e.g.:
    "Flagged due to: unusual geo-velocity (78,432 km/h, +4.2 SHAP),
     new device fingerprint (+2.1 SHAP), off-hours access (+0.8 SHAP)"

This directly satisfies deliverable #5 (explainability layer / feature
attribution per alert) and feeds the dashboard's "contributing factors" view.
"""

import pandas as pd
import os
import numpy as np
import shap
import joblib
from model_io import load_model

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import FEATURE_COLUMNS, ensure_graph_features

MODEL_PATH = f"{_PROJECT_ROOT}/models/xgb_detector.joblib"
ENCODER_PATH = f"{_PROJECT_ROOT}/models/label_encoder.joblib"
DATA_PATH = f"{_PROJECT_ROOT}/data/access_logs_featured.csv"

# Human-readable labels + units for the top features, used to build rationale text
FEATURE_DISPLAY = {
    "geo_velocity_kmh": ("implied travel velocity", "km/h"),
    "time_since_last_session_min": ("time since entity's last session", "min"),
    "is_new_resource_for_entity": ("access to a resource never used before", ""),
    "is_new_device_for_entity": ("unrecognized device fingerprint", ""),
    "hour_deviation_zscore": ("deviation from entity's typical login hour", "std devs"),
    "duration_zscore": ("deviation from entity's typical session length", "std devs"),
    "is_cold_start_entity": ("entity has little/no prior history (cold start)", ""),
    "distinct_entities_per_ip_1h": ("distinct accounts from same source IP (1h)", "accounts"),
    "failed_auth_count_ip_1h": ("failed logins from this source IP (1h)", "attempts"),
    "failed_auth_count_entity_24h": ("failed logins for this entity (24h)", "attempts"),
    "is_off_hours": ("access occurred outside normal hours", ""),
    "is_privileged_resource": ("access to a privileged/sensitive resource", ""),
    "command_seq_length": ("number of privileged actions in session", "actions"),
    "is_failure": ("authentication failed", ""),
    "entity_session_seq_num": ("entity's session history depth", "sessions"),
    "day_of_week": ("day of week", ""),
    "hour_of_day": ("hour of day", ""),
}


def build_rationale(row, shap_row, feature_cols, top_n=3):
    """Return a short human-readable rationale string from the top-N
    contributing SHAP features for this single prediction."""
    contribs = list(zip(feature_cols, shap_row))
    contribs.sort(key=lambda x: -abs(x[1]))
    parts = []
    for feat, val in contribs[:top_n]:
        if abs(val) < 0.05:
            continue
        label, unit = FEATURE_DISPLAY.get(feat, (feat, ""))
        raw_val = row[feat]
        if unit:
            parts.append(f"{label} ({raw_val:.1f} {unit}, SHAP {val:+.2f})")
        else:
            parts.append(f"{label} (SHAP {val:+.2f})")
    if not parts:
        return "No single dominant factor; flagged on combined weak signals."
    return "Flagged due to: " + "; ".join(parts)


def main(sample_size=500):
    ensure_graph_features()
    model, le = load_model()
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df[df["label"] != "edge_case_insider_drift"].copy()

    # Focus explainability compute on the highest-risk rows (top N by predicted anomaly prob)
    # to keep this fast -- full-dataset SHAP is unnecessary for a dashboard alert queue.
    proba = model.predict_proba(df[FEATURE_COLUMNS])
    normal_idx = list(le.classes_).index("normal")
    df["anomaly_score"] = 1 - proba[:, normal_idx]
    df["predicted_label"] = le.inverse_transform(model.predict(df[FEATURE_COLUMNS]))

    top_alerts = df.sort_values("anomaly_score", ascending=False).head(sample_size).copy()

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(top_alerts[FEATURE_COLUMNS])
    # shap_values shape for multiclass: (n_classes, n_samples, n_features) in older API,
    # or (n_samples, n_features, n_classes) in newer -- normalize to per-sample list
    shap_values = np.array(shap_values)
    if shap_values.ndim == 3 and shap_values.shape[0] == len(le.classes_):
        # (n_classes, n_samples, n_features) -> pick predicted class's shap row per sample
        pred_class_idx = model.predict(top_alerts[FEATURE_COLUMNS])
        per_sample_shap = np.array([shap_values[c, i, :] for i, c in enumerate(pred_class_idx)])
    else:
        pred_class_idx = model.predict(top_alerts[FEATURE_COLUMNS])
        per_sample_shap = np.array([shap_values[i, :, c] for i, c in enumerate(pred_class_idx)])

    rationales = []
    for (idx, row), shap_row in zip(top_alerts.iterrows(), per_sample_shap):
        rationales.append(build_rationale(row, shap_row, FEATURE_COLUMNS))
    top_alerts["rationale"] = rationales

    out_cols = ["session_id", "entity_id", "entity_type", "timestamp", "resource_accessed",
                "source_ip", "label", "predicted_label", "anomaly_score", "rationale"]
    top_alerts[out_cols].to_csv(f"{_PROJECT_ROOT}/reports/alert_queue.csv", index=False)
    print(f"Saved {len(top_alerts)} explained alerts to reports/alert_queue.csv")
    print("\nSample rationales:")
    for r in top_alerts["rationale"].head(5):
        print(" -", r)


if __name__ == "__main__":
    main()
