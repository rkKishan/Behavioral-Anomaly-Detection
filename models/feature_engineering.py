"""
Feature Engineering for Behavioral Anomaly Detection
======================================================

All features are computed CAUSALLY (only using data up to and including the
current row's timestamp per entity/IP) to avoid label leakage from future
sessions -- this matters for a realistic real-time detection simulation.

Feature groups:
  1. Temporal        - hour, day-of-week, time-since-last-session
  2. Entity-behavior  - resource novelty, device novelty, session-duration deviation
  3. Geo-velocity     - implied travel speed vs. last known location (impossible travel signal)
  4. Network-level    - distinct entities per source IP, failed-auth bursts (brute force /
                        credential stuffing signals)
  5. Cold-start flag  - marks entities with < MIN_HISTORY prior sessions
"""

import os
import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

MIN_HISTORY = 8  # sessions needed before an entity is no longer "cold start"
EARTH_R_KM = 6371.0


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(a))


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values(["entity_id", "timestamp"]).reset_index(drop=True)

    df["hour_of_day"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60.0
    df["day_of_week"] = df["timestamp"].dt.dayofweek

    # --- Per-entity causal rolling features ---
    prev_ts, prev_lat, prev_lon, prev_devices, prev_resources = {}, {}, {}, {}, {}
    entity_hour_history = {}
    entity_duration_history = {}
    session_count_hist = {}

    transition_counts = {}   # eid -> {prev_resource: {curr_resource: count}}
    prev_resource_seen = {}  # eid -> last resource accessed
    seq_surprise = np.zeros(len(df))
    time_since_last = np.zeros(len(df))
    geo_velocity_kmh = np.zeros(len(df))
    is_new_resource = np.zeros(len(df), dtype=int)
    is_new_device = np.zeros(len(df), dtype=int)
    hour_deviation = np.zeros(len(df))
    duration_zscore = np.zeros(len(df))
    is_cold_start = np.zeros(len(df), dtype=int)
    entity_session_seq_num = np.zeros(len(df), dtype=int)

    for i, row in df.iterrows():
        eid = row["entity_id"]
        ts = row["timestamp"]

        # session sequence number / cold start
        n_prior = session_count_hist.get(eid, 0)
        entity_session_seq_num[i] = n_prior
        is_cold_start[i] = 1 if n_prior < MIN_HISTORY else 0

        # time since last session (minutes); large default for first-ever session
        if eid in prev_ts:
            delta_min = (ts - prev_ts[eid]).total_seconds() / 60.0
            time_since_last[i] = delta_min
        else:
            time_since_last[i] = 999999.0  # sentinel for "no history"

        # geo velocity vs last known location
        if eid in prev_lat and pd.notna(row.get("geo_lat")) and pd.notna(prev_lat[eid]):
            dist = haversine_km(prev_lat[eid], prev_lon[eid], row["geo_lat"], row["geo_lon"])
            hours_gap = max(time_since_last[i] / 60.0, 1e-6)
            geo_velocity_kmh[i] = dist / hours_gap
        else:
            geo_velocity_kmh[i] = 0.0

        # resource / device novelty
        seen_resources = prev_resources.get(eid, set())
        is_new_resource[i] = 0 if (row["resource_accessed"] in seen_resources or n_prior < 2) else 1
        seen_devices = prev_devices.get(eid, set())
        is_new_device[i] = 0 if (row["device_fingerprint"] in seen_devices or n_prior < 2) else 1

        # hour-of-day deviation from entity's historical mean login hour
        hist_hours = entity_hour_history.get(eid, [])
        if len(hist_hours) >= 3:
            mu, sigma = np.mean(hist_hours), max(np.std(hist_hours), 0.5)
            hour_deviation[i] = abs(row["hour_of_day"] - mu) / sigma
        else:
            hour_deviation[i] = 0.0

        # session-duration z-score vs entity history
        hist_durations = entity_duration_history.get(eid, [])
        if len(hist_durations) >= 3:
            mu_d, sigma_d = np.mean(hist_durations), max(np.std(hist_durations), 0.5)
            duration_zscore[i] = (row["session_duration"] - mu_d) / sigma_d
        else:
            duration_zscore[i] = 0.0

        # Sequence-aware surprise: -log P(curr_resource | prev_resource) under
        # the entity's own causal bigram model (Laplace-smoothed). High surprise =
        # a transition this entity has never made = lateral-movement-style signal.
        prev_r = prev_resource_seen.get(eid)
        if prev_r is not None:
            trans = transition_counts.get(eid, {}).get(prev_r, {})
            total = sum(trans.values())
            count = trans.get(row["resource_accessed"], 0)
            p = (count + 1) / (total + 20)  # Laplace smoothing over ~20 resources
            seq_surprise[i] = -np.log(p)
        else:
            seq_surprise[i] = 0.0

        # ---- update rolling history AFTER computing features (causal) ----
        if prev_r is not None:
            transition_counts.setdefault(eid, {}).setdefault(prev_r, {})
            transition_counts[eid][prev_r][row["resource_accessed"]] = \
                transition_counts[eid][prev_r].get(row["resource_accessed"], 0) + 1
        prev_resource_seen[eid] = row["resource_accessed"]
        prev_ts[eid] = ts
        if pd.notna(row.get("geo_lat")):
            prev_lat[eid], prev_lon[eid] = row["geo_lat"], row["geo_lon"]
        prev_resources.setdefault(eid, set()).add(row["resource_accessed"])
        prev_devices.setdefault(eid, set()).add(row["device_fingerprint"])
        # Concept-drift handling: histories are ROLLING windows (last 30 obs),
        # so the learned "normal" adapts as legitimate behavior evolves and
        # old patterns age out instead of being flagged forever.
        DRIFT_WINDOW = 30
        hh = entity_hour_history.setdefault(eid, []); hh.append(row["hour_of_day"])
        if len(hh) > DRIFT_WINDOW: del hh[:-DRIFT_WINDOW]
        dh = entity_duration_history.setdefault(eid, []); dh.append(row["session_duration"])
        if len(dh) > DRIFT_WINDOW: del dh[:-DRIFT_WINDOW]
        session_count_hist[eid] = n_prior + 1

    df["time_since_last_session_min"] = time_since_last
    df["resource_seq_surprise"] = seq_surprise
    df["geo_velocity_kmh"] = geo_velocity_kmh
    df["is_new_resource_for_entity"] = is_new_resource
    df["is_new_device_for_entity"] = is_new_device
    df["hour_deviation_zscore"] = hour_deviation
    df["duration_zscore"] = duration_zscore
    df["is_cold_start_entity"] = is_cold_start
    df["entity_session_seq_num"] = entity_session_seq_num

    # --- Network-level features (per source_ip, rolling 1-hour windows) ---
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["is_failure"] = (df["auth_result"] == "failure").astype(int)

    df_idx = df.set_index("timestamp")
    # distinct entities per source_ip in trailing 1h (credential stuffing signal)
    distinct_entities_1h = []
    failed_count_ip_1h = []
    ip_window = {}  # source_ip -> list of (timestamp, entity_id, is_failure)

    for ts, row in zip(df["timestamp"], df.to_dict("records")):
        ip = row["source_ip"]
        window = ip_window.setdefault(ip, [])
        # drop entries older than 1h
        cutoff = ts - pd.Timedelta(hours=1)
        window[:] = [w for w in window if w[0] >= cutoff]
        distinct_entities_1h.append(len({w[1] for w in window}))
        failed_count_ip_1h.append(sum(w[2] for w in window))
        window.append((ts, row["entity_id"], row["is_failure"]))

    df["distinct_entities_per_ip_1h"] = distinct_entities_1h
    df["failed_auth_count_ip_1h"] = failed_count_ip_1h

    # failed-auth count for the entity itself, trailing 24h
    entity_fail_window = {}
    failed_count_entity_24h = []
    for ts, row in zip(df["timestamp"], df.to_dict("records")):
        eid = row["entity_id"]
        window = entity_fail_window.setdefault(eid, [])
        cutoff = ts - pd.Timedelta(hours=24)
        window[:] = [w for w in window if w[0] >= cutoff]
        failed_count_entity_24h.append(sum(w[1] for w in window))
        window.append((ts, row["is_failure"]))
    df["failed_auth_count_entity_24h"] = failed_count_entity_24h

    df["is_off_hours"] = df["hour_of_day"].apply(lambda h: 1 if (h < 6 or h > 22) else 0)
    PRIVILEGED_RESOURCES = {
        "db_prod_customers", "db_prod_orders", "admin_panel_iam",
        "admin_panel_billing", "financial_system_erp", "iot_hub_controller",
    }
    df["is_privileged_resource"] = df["resource_accessed"].isin(PRIVILEGED_RESOURCES).astype(int)
    df["command_seq_length"] = df["command_sequence"].fillna("").apply(
        lambda s: 0 if s == "" else len(s.split(";")))

    return df


GRAPH_FEATURES = ["graph_edge_weight", "resource_popularity", "peer_affinity",
                  "two_hop_reachable", "graph_anomaly_score"]


def ensure_graph_features(csv_path=None):
    """Self-healing pipeline guard.

    The graph-based detector (models/graph_model.py) appends its features to
    access_logs_featured.csv. If a downstream step runs before it, every
    later script would otherwise die with an opaque KeyError. This checks for
    the graph columns and builds them on demand, so the steps can be run in
    any order (or individually) without manual sequencing.
    """
    import pandas as _pd
    path = csv_path or f"{_PROJECT_ROOT}/data/access_logs_featured.csv"
    try:
        cols = _pd.read_csv(path, nrows=1).columns
    except FileNotFoundError:
        raise SystemExit(
            "\nERROR: data/access_logs_featured.csv not found.\n"
            "Run these first:\n"
            "  python data/generate_synthetic_logs.py\n"
            "  python models/feature_engineering.py\n")
    if all(c in cols for c in GRAPH_FEATURES):
        return False
    print("[pipeline] graph features missing -> running graph model first...")
    import graph_model
    graph_model.main()
    print("[pipeline] graph features ready.\n")
    return True


FEATURE_COLUMNS = [
    "hour_of_day", "day_of_week", "time_since_last_session_min", "geo_velocity_kmh",
    "is_new_resource_for_entity", "is_new_device_for_entity", "hour_deviation_zscore",
    "duration_zscore", "is_cold_start_entity", "entity_session_seq_num",
    "distinct_entities_per_ip_1h", "failed_auth_count_ip_1h", "failed_auth_count_entity_24h",
    "is_off_hours", "is_privileged_resource", "command_seq_length", "is_failure",
    "resource_seq_surprise",
    # graph-based entity-resource relationship features (models/graph_model.py)
    "graph_edge_weight", "resource_popularity", "peer_affinity",
    "two_hop_reachable", "graph_anomaly_score",
]

if __name__ == "__main__":
    df = pd.read_csv(f"{_PROJECT_ROOT}/data/access_logs_labeled.csv")
    feats = build_features(df)
    feats.to_csv(f"{_PROJECT_ROOT}/data/access_logs_featured.csv", index=False)
    print(f"Feature engineering complete. Shape: {feats.shape}")
    # graph_* columns are appended later by models/graph_model.py, so only
    # summarise the columns that exist at this stage of the pipeline
    present = [c for c in FEATURE_COLUMNS if c in feats.columns]
    print(feats[present].describe().T[["mean", "std", "min", "max"]])
