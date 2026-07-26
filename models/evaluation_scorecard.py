"""
Evaluation Scorecard
=======================

Maps each of the problem statement's seven evaluation criteria to a measured
number from this pipeline, so a reviewer can verify claims directly rather
than taking them on trust. Writes reports/evaluation_scorecard.json.
"""

import json
import os
import time

import joblib
from model_io import load_model
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

from feature_engineering import FEATURE_COLUMNS, ensure_graph_features

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BUDGET = 0.01


def main():
    ensure_graph_features()
    model, le = load_model()
    df = pd.read_csv(f"{_PROJECT_ROOT}/data/access_logs_featured.csv",
                     parse_dates=["timestamp"]).sort_values("timestamp")
    split = df["timestamp"].quantile(0.75)
    test = df[df["timestamp"] > split].reset_index(drop=True)
    scored = test[test["label"] != "edge_case_insider_drift"].reset_index(drop=True)

    proba = model.predict_proba(scored[FEATURE_COLUMNS])
    normal_idx = list(le.classes_).index("normal")
    score = 1 - proba[:, normal_idx]
    y_true = (scored["label"] != "normal").astype(int).values
    pred = le.inverse_transform(model.predict(scored[FEATURE_COLUMNS]))

    sc = {}

    # ---- 1. Detection accuracy on IMBALANCED labels ----
    # Accuracy is meaningless at 2% prevalence; PR-AUC is the correct metric.
    prevalence = y_true.mean()
    sc["1_detection_accuracy_imbalanced"] = {
        "anomaly_prevalence": float(prevalence),
        "imbalance_ratio": f"1:{int((1-prevalence)/max(prevalence,1e-9))}",
        "pr_auc_average_precision": float(average_precision_score(y_true, score)),
        "pr_auc_baseline_random": float(prevalence),
        "roc_auc": float(roc_auc_score(y_true, score)),
        "note": ("PR-AUC is reported because accuracy is uninformative at this "
                 "prevalence: always predicting 'normal' would score ~98%."),
    }

    # ---- 2. Correct anomaly-TYPE classification ----
    anom = scored[scored["label"] != "normal"]
    anom_pred = pred[(scored["label"] != "normal").values]
    type_acc = float((anom_pred == anom["label"].values).mean())
    per_class = {}
    for c in sorted(anom["label"].unique()):
        m = anom["label"].values == c
        per_class[c] = {
            "support": int(m.sum()),
            "correct_type_rate": float((anom_pred[m] == c).mean()),
        }
    sc["2_anomaly_type_classification"] = {
        "overall_correct_type_given_detected": type_acc,
        "macro_f1_all_classes": float(f1_score(scored["label"], pred, average="macro")),
        "per_class": per_class,
    }

    # ---- 3. FALSE POSITIVE RATE at a realistic analyst alert budget ----
    k = max(1, int(BUDGET * len(scored)))
    order = np.argsort(-score)
    alert_idx = order[:k]
    fp = int((y_true[alert_idx] == 0).sum())
    tp = int((y_true[alert_idx] == 1).sum())
    benign_total = int((y_true == 0).sum())
    days = (test["timestamp"].max() - test["timestamp"].min()).days or 1
    sc["3_false_positives_at_alert_budget"] = {
        "budget_fraction_of_events": BUDGET,
        "alerts_issued": int(k),
        "true_positives": tp,
        "false_positives": fp,
        "precision_at_budget": float(tp / k),
        "false_positive_rate_vs_all_benign": float(fp / benign_total),
        "recall_at_budget": float(tp / max(y_true.sum(), 1)),
        "analyst_workload_alerts_per_day": float(k / days),
        "false_alarms_per_day": float(fp / days),
    }

    # ---- 4. Explainability / analyst usability ----
    aq = pd.read_csv(f"{_PROJECT_ROOT}/reports/alert_queue.csv")
    sc["4_explainability_usability"] = {
        "alerts_with_rationale": int(aq["rationale"].notna().sum()),
        "pct_alerts_with_rationale": float(aq["rationale"].notna().mean()),
        "rationale_names_specific_features": bool(
            aq["rationale"].str.contains("SHAP").mean() > 0.9),
        "dashboard_views": ["ranked alert queue", "risk score", "contributing factors",
                            "entity history", "robustness (cold-start & drift)"],
        "example_rationale": str(aq["rationale"].iloc[0])[:180],
    }

    # ---- 5. Cold-start & concept drift ----
    with open(f"{_PROJECT_ROOT}/reports/robustness_metrics.json") as f:
        rob = json.load(f)
    sc["5_cold_start_and_drift"] = {
        "cold_start_false_positive_rate": rob["cold_start"]["false_positive_rate"],
        "cold_start_entities": rob["cold_start"]["new_entities"],
        "cold_start_benign_sessions": rob["cold_start"]["benign_sessions_from_new_entities"],
        "drift_false_positive_rate": rob["concept_drift"]["false_positive_rate"],
        "drift_entities": rob["concept_drift"]["drifted_entities"],
        "drift_post_change_sessions": rob["concept_drift"]["post_drift_benign_sessions"],
        "profile_adaptation_evidence": {
            "post_drift_hour_deviation_z": rob["concept_drift"].get(
                "mean_hour_deviation_zscore_after_drift"),
            "unchanged_entities_hour_deviation_z": rob["concept_drift"].get(
                "mean_hour_deviation_zscore_baseline_cohort"),
        },
    }

    # ---- 6. System design & scalability (streaming feasibility) ----
    N = min(2000, len(scored))
    feats = scored[FEATURE_COLUMNS]
    t0 = time.perf_counter()
    for i in range(N):
        model.predict_proba(feats.iloc[[i]])
    per_event = (time.perf_counter() - t0) / N

    B = min(20000, len(scored))
    t0 = time.perf_counter()
    model.predict_proba(feats.iloc[:B])
    batch_elapsed = time.perf_counter() - t0

    sc["6_scalability_streaming"] = {
        "single_event_latency_ms": per_event * 1000,
        "single_threaded_events_per_sec": 1 / per_event,
        "batched_events_per_sec": B / batch_elapsed,
        "feature_computation": "causal/stateful -> portable to Kafka + per-entity state store",
        "honest_note": ("Per-event latency suits real-time scoring. Sustained "
                        "high-volume ingest would use micro-batching (measured "
                        "above) and horizontal partitioning by entity_id, since "
                        "all per-entity state is independent."),
    }

    # ---- 7. Report clarity ----
    readme = open(f"{_PROJECT_ROOT}/reports/README.md").read()
    sc["7_report_clarity"] = {
        "sections": [l.strip("# ").strip() for l in readme.splitlines()
                     if l.startswith("## ")],
        "documents_limitations": "Known Limitations" in readme,
        "documents_assumptions": "Assumptions" in readme,
        "word_count": len(readme.split()),
    }

    with open(f"{_PROJECT_ROOT}/reports/evaluation_scorecard.json", "w") as f:
        json.dump(sc, f, indent=2)

    # ---- print ----
    W = 70
    print("=" * W)
    print("EVALUATION CRITERIA SCORECARD".center(W))
    print("=" * W)
    c1 = sc["1_detection_accuracy_imbalanced"]
    print(f"\n1. Detection accuracy on imbalanced labels")
    print(f"   imbalance {c1['imbalance_ratio']} | PR-AUC {c1['pr_auc_average_precision']:.3f} "
          f"(random baseline {c1['pr_auc_baseline_random']:.3f}) | ROC-AUC {c1['roc_auc']:.3f}")

    c2 = sc["2_anomaly_type_classification"]
    print(f"\n2. Correct anomaly-type classification")
    print(f"   correct type on detected anomalies: {c2['overall_correct_type_given_detected']:.1%}"
          f" | macro-F1 {c2['macro_f1_all_classes']:.3f}")

    c3 = sc["3_false_positives_at_alert_budget"]
    print(f"\n3. False positives at top-{BUDGET:.0%} analyst alert budget")
    print(f"   {c3['alerts_issued']} alerts -> {c3['true_positives']} true / "
          f"{c3['false_positives']} false | precision {c3['precision_at_budget']:.1%}")
    print(f"   FP rate vs all benign traffic: {c3['false_positive_rate_vs_all_benign']:.4%}"
          f" | {c3['false_alarms_per_day']:.1f} false alarms/day")

    c5 = sc["5_cold_start_and_drift"]
    print(f"\n5. Cold-start & concept drift")
    print(f"   new entities  : {c5['cold_start_false_positive_rate']:.2%} FP "
          f"({c5['cold_start_benign_sessions']:,} sessions)")
    print(f"   after drift   : {c5['drift_false_positive_rate']:.2%} FP "
          f"({c5['drift_post_change_sessions']:,} sessions)")

    c6 = sc["6_scalability_streaming"]
    print(f"\n6. Scalability / streaming")
    print(f"   {c6['single_event_latency_ms']:.2f} ms per event "
          f"({c6['single_threaded_events_per_sec']:,.0f}/sec single-thread)")
    print(f"   batched: {c6['batched_events_per_sec']:,.0f} events/sec")

    print(f"\nSaved reports/evaluation_scorecard.json")


if __name__ == "__main__":
    main()
