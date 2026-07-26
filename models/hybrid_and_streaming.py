"""
Hybrid Scoring (ML + rule layer)  &  Streaming Feasibility Demo
==================================================================

1. RULE LAYER: physics doesn't need training data. Any session whose implied
   travel velocity exceeds 900 km/h (max commercial flight speed) vs. the
   entity's previous location is deterministically classified as
   impossible_travel, overriding the ML model. This fixes the one class the
   ML under-learns from rarity, exactly as real SOC systems layer detections.

2. STREAMING DEMO: replays the held-out test window event-by-event through
   the full scoring path (model + rule layer) as a stateless per-event loop,
   measuring per-event latency -- demonstrating real-time feasibility.
   (Features are already computed causally/statefully, so the same logic
   ports directly to a Kafka-style stateful stream processor.)
"""

import time
import json
import os
import numpy as np
import pandas as pd
import joblib
from model_io import load_model
from sklearn.metrics import classification_report

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import FEATURE_COLUMNS, ensure_graph_features

GEO_VELOCITY_LIMIT_KMH = 900.0


def apply_rule_layer(pred_labels, features_df, known_cities=None):
    """Deterministic overrides on top of ML predictions.

    Fires only when BOTH hold:
      (a) implied travel velocity exceeds commercial flight speed, AND
      (b) the session originates from a city the entity has never used.

    Condition (b) suppresses the "return trip" artifact: immediately after a
    genuine impossible-travel event, the entity's next *legitimate* home-city
    login also shows a high implied velocity relative to the attacker's remote
    location. Without (b) the rule double-counts a single incident and
    penalises the innocent follow-up session.
    """
    out = np.asarray(pred_labels).copy()
    fast = features_df["geo_velocity_kmh"].values > GEO_VELOCITY_LIMIT_KMH
    # Spec defines impossible travel as an entity "logging in from
    # geographically distant locations" -- i.e. SUCCESSFUL authentications.
    # A failed burst from a foreign host is brute force / credential stuffing,
    # which also trips a raw velocity check, so success is required here to
    # keep the two taxonomies separate.
    succeeded = (features_df["auth_result"].values == "success")
    fast = fast & succeeded
    if known_cities is not None:
        unfamiliar = np.array([
            city not in known_cities.get(eid, set())
            for eid, city in zip(features_df["entity_id"], features_df["geo_city"])
        ])
        override_mask = fast & unfamiliar
    else:
        override_mask = fast
    out[override_mask] = "anomaly_impossible_travel"

    # VETO: impossible travel is defined by physics, so a prediction of that
    # class that violates no physical constraint cannot be correct. Such
    # predictions are downgraded to the model's next-best class.
    veto_mask = (out == "anomaly_impossible_travel") & (~override_mask)
    return out, int(override_mask.sum()), veto_mask


def main():
    ensure_graph_features()
    model, le = load_model()
    df = pd.read_csv(f"{_PROJECT_ROOT}/data/access_logs_featured.csv",
                     parse_dates=["timestamp"]).sort_values("timestamp")
    df = df[df["label"] != "edge_case_insider_drift"].reset_index(drop=True)
    split_ts = df["timestamp"].quantile(0.75)
    test = df[df["timestamp"] > split_ts].reset_index(drop=True)

    # Cities each entity legitimately used during TRAINING only (no leakage)
    train = df[df["timestamp"] <= split_ts]
    known_cities = (train[train["label"] == "normal"]
                    .groupby("entity_id")["geo_city"]
                    .apply(lambda s: set(s.unique())).to_dict())

    # ---- Hybrid evaluation (batch) ----
    proba = model.predict_proba(test[FEATURE_COLUMNS])
    ml_pred = le.inverse_transform(model.predict(test[FEATURE_COLUMNS]))
    hybrid_pred, n_overrides, veto_mask = apply_rule_layer(ml_pred, test, known_cities)

    # Resolve vetoed rows to the next-most-likely class
    it_idx = list(le.classes_).index("anomaly_impossible_travel")
    if veto_mask.any():
        alt = proba.copy()
        alt[:, it_idx] = -1.0
        hybrid_pred[veto_mask] = le.inverse_transform(alt[veto_mask].argmax(axis=1))
    print(f"Rule layer CONFIRMED {n_overrides} sessions (geo-velocity > "
          f"{GEO_VELOCITY_LIMIT_KMH} km/h from an unfamiliar city)")
    print(f"Rule layer VETOED {int(veto_mask.sum())} physically-impossible "
          f"model predictions (downgraded to next-best class)")
    print("\n=== HYBRID (ML + rule layer) classification report ===")
    rep = classification_report(test["label"], hybrid_pred, zero_division=0, output_dict=True)
    print(classification_report(test["label"], hybrid_pred, zero_division=0))

    with open(f"{_PROJECT_ROOT}/reports/hybrid_metrics.json", "w") as f:
        json.dump({"rule_overrides": n_overrides,
                   "geo_velocity_limit_kmh": GEO_VELOCITY_LIMIT_KMH,
                   "classification_report": rep}, f, indent=2)

    # ---- Streaming feasibility: per-event scoring latency ----
    N = min(2000, len(test))
    sample = test.head(N)
    feats = sample[FEATURE_COLUMNS]
    t0 = time.perf_counter()
    for i in range(N):
        row = feats.iloc[[i]]                       # single event arrives
        proba = model.predict_proba(row)            # ML score
        lbl = le.inverse_transform([int(np.argmax(proba))])  # class
        eid = sample["entity_id"].iloc[i]
        if (row["geo_velocity_kmh"].iloc[0] > GEO_VELOCITY_LIMIT_KMH
                and sample["geo_city"].iloc[i] not in known_cities.get(eid, set())):
            lbl = ["anomaly_impossible_travel"]  # rule layer
    elapsed = time.perf_counter() - t0
    per_event_ms = elapsed / N * 1000
    throughput = N / elapsed
    print(f"\n=== STREAMING FEASIBILITY ===")
    print(f"Scored {N} events one-by-one: {per_event_ms:.2f} ms/event  "
          f"({throughput:,.0f} events/sec single-threaded)")
    print("Feature computation is causal/stateful by design -> direct port to "
          "stream processing (Kafka consumer + per-entity state store).")

    with open(f"{_PROJECT_ROOT}/reports/streaming_benchmark.json", "w") as f:
        json.dump({"events_scored": N, "ms_per_event": per_event_ms,
                   "events_per_sec_single_thread": throughput}, f, indent=2)


if __name__ == "__main__":
    main()
