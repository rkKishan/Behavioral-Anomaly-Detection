"""
Baseline Profiling Model  (Deliverable #2)
=============================================

A distinct per-entity "normal behaviour" representation, separate from the
supervised detection model. Two complementary components:

1. STATISTICAL PROFILE (per entity)
   For every entity, computed ONLY from the training time window:
     - login-hour mean/std, session-duration mean/std
     - known resource set, known device fingerprints, home geo centroid
   Any session can be scored against its entity's profile as a weighted sum
   of z-scores + novelty indicators -> interpretable "profile deviation score".

2. ONE-CLASS MODEL (IsolationForest)
   Trained ONLY on normal training sessions' behavioural features -- learns
   the joint shape of "normal" without ever seeing an attack label. Provides
   an unsupervised anomaly score that does not depend on the attack taxonomy,
   which is exactly what catches *novel* attack types the supervised model
   was never trained on.

COLD-START HANDLING (explicit):
   Entities with < MIN_HISTORY training sessions get a POPULATION-LEVEL
   profile for their entity_type (user / service_account / edge_device)
   instead of an unreliable personal one, and their sessions carry the
   cold-start flag so the analyst dashboard can surface the reduced
   confidence.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.metrics import roc_auc_score

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import FEATURE_COLUMNS, ensure_graph_features

DATA_PATH = f"{_PROJECT_ROOT}/data/access_logs_featured.csv"
MIN_HISTORY = 8


def build_statistical_profiles(train_df):
    """Per-entity profile + per-entity-type population fallback profiles."""
    profiles = {}
    normal = train_df[train_df["label"] == "normal"]

    # population-level fallbacks per entity_type (cold-start support)
    population = {}
    for etype, grp in normal.groupby("entity_type"):
        population[etype] = {
            "hour_mean": float(grp["hour_of_day"].mean()),
            "hour_std": float(max(grp["hour_of_day"].std(), 0.5)),
            "dur_mean": float(grp["session_duration"].mean()),
            "dur_std": float(max(grp["session_duration"].std(), 0.5)),
        }

    for eid, grp in normal.groupby("entity_id"):
        if len(grp) >= MIN_HISTORY:
            profiles[eid] = {
                "cold_start": False,
                "hour_mean": float(grp["hour_of_day"].mean()),
                "hour_std": float(max(grp["hour_of_day"].std(), 0.5)),
                "dur_mean": float(grp["session_duration"].mean()),
                "dur_std": float(max(grp["session_duration"].std(), 0.5)),
                "known_resources": sorted(grp["resource_accessed"].unique().tolist()),
                "known_devices": sorted(grp["device_fingerprint"].unique().tolist()),
            }
        else:
            etype = grp["entity_type"].iloc[0]
            pop = population[etype]
            profiles[eid] = {"cold_start": True, **pop,
                             "known_resources": [], "known_devices": []}
    return profiles, population


def profile_deviation_score(row, profile):
    """Interpretable weighted deviation of one session from an entity profile."""
    hour_z = abs(row["hour_of_day"] - profile["hour_mean"]) / profile["hour_std"]
    dur_z = abs(row["session_duration"] - profile["dur_mean"]) / profile["dur_std"]
    new_resource = 0.0
    new_device = 0.0
    if profile["known_resources"]:
        new_resource = 0.0 if row["resource_accessed"] in profile["known_resources"] else 1.0
    if profile["known_devices"]:
        new_device = 0.0 if row["device_fingerprint"] in profile["known_devices"] else 1.0
    failed = 1.0 if row["auth_result"] == "failure" else 0.0
    return 0.8 * hour_z + 0.5 * dur_z + 2.0 * new_resource + 3.0 * new_device + 2.0 * failed


def main():
    ensure_graph_features()
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
    df = df[df["label"] != "edge_case_insider_drift"].reset_index(drop=True)
    split_ts = df["timestamp"].quantile(0.75)
    train_df = df[df["timestamp"] <= split_ts]
    test_df = df[df["timestamp"] > split_ts].copy()

    # --- Component 1: statistical profiles ---
    profiles, population = build_statistical_profiles(train_df)
    default_pop = population["user"]

    def get_profile(eid, etype):
        if eid in profiles:
            return profiles[eid]
        return {"cold_start": True, **population.get(etype, default_pop),
                "known_resources": [], "known_devices": []}

    test_df["profile_deviation_score"] = [
        profile_deviation_score(row, get_profile(row["entity_id"], row["entity_type"]))
        for row in test_df.to_dict("records")
    ]

    # --- Component 2: IsolationForest one-class model (normal train data only) ---
    iso = IsolationForest(n_estimators=200, contamination="auto", random_state=42, n_jobs=-1)
    normal_train = train_df[train_df["label"] == "normal"]
    iso.fit(normal_train[FEATURE_COLUMNS])
    # decision_function: higher = more normal -> invert to anomaly score
    test_df["iforest_anomaly_score"] = -iso.decision_function(test_df[FEATURE_COLUMNS])

    # --- Evaluate both baseline components alone (binary: anomaly vs normal) ---
    y_true = (test_df["label"] != "normal").astype(int)
    auc_profile = roc_auc_score(y_true, test_df["profile_deviation_score"])
    auc_iforest = roc_auc_score(y_true, test_df["iforest_anomaly_score"])
    print(f"Baseline statistical-profile score AUC: {auc_profile:.3f}")
    print(f"Baseline IsolationForest score AUC:     {auc_iforest:.3f}")
    n_cold = sum(1 for r in test_df.to_dict("records")
                 if get_profile(r["entity_id"], r["entity_type"])["cold_start"])
    print(f"Cold-start sessions scored via population fallback: {n_cold}")

    # --- Persist ---
    with open(f"{_PROJECT_ROOT}/models/entity_profiles.json", "w") as f:
        json.dump({"profiles": profiles, "population_fallbacks": population}, f, indent=1)
    joblib.dump(iso, f"{_PROJECT_ROOT}/models/iforest_baseline.joblib")
    test_df[["session_id", "profile_deviation_score", "iforest_anomaly_score"]].to_csv(
        f"{_PROJECT_ROOT}/reports/baseline_scores.csv", index=False)

    with open(f"{_PROJECT_ROOT}/reports/baseline_model_metrics.json", "w") as f:
        json.dump({"statistical_profile_auc": auc_profile,
                   "isolation_forest_auc": auc_iforest,
                   "cold_start_sessions_in_test": int(n_cold),
                   "entities_profiled": len(profiles)}, f, indent=2)
    print("Saved entity_profiles.json, iforest_baseline.joblib, baseline metrics")


if __name__ == "__main__":
    main()
