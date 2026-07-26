"""
Detection Model Training
==========================

Approach: XGBoost multiclass classifier over engineered behavioral features.
Chosen over LSTM/Transformer as the primary model because:
  - Faster to train/iterate within hackathon time constraints
  - Natively supports SHAP explainability (required deliverable)
  - Handles tabular + engineered sequence-derived features well
  - Class imbalance handled via inverse-frequency sample weighting

Evaluation uses a TIME-BASED split (not random) -- train on first ~75% of
days, test on the last ~25% -- to realistically simulate deployment where
the model must generalize to future sessions, not just held-out random rows.
"""

import pandas as pd
import os
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import joblib
from model_io import save_model
import json

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

from feature_engineering import FEATURE_COLUMNS, ensure_graph_features

DATA_PATH = f"{_PROJECT_ROOT}/data/access_logs_featured.csv"
MODEL_OUT = f"{_PROJECT_ROOT}/models/xgb_detector.joblib"
ENCODER_OUT = f"{_PROJECT_ROOT}/models/label_encoder.joblib"
METRICS_OUT = f"{_PROJECT_ROOT}/reports/model_metrics.json"


def main():
    ensure_graph_features()
    df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Binary "insider_drift" edge case: exclude from the main multiclass
    # detection task (it's ambiguous by design -- used later for FP tuning
    # discussion in the report) but keep normal + true anomalies.
    df_model = df[df["label"] != "edge_case_insider_drift"].copy()

    le = LabelEncoder()
    df_model["label_enc"] = le.fit_transform(df_model["label"])

    # Time-based split: last 25% of the timeline is the test set
    split_ts = df_model["timestamp"].quantile(0.75)
    train_df = df_model[df_model["timestamp"] <= split_ts]
    test_df = df_model[df_model["timestamp"] > split_ts]

    X_train, y_train = train_df[FEATURE_COLUMNS], train_df["label_enc"]
    X_test, y_test = test_df[FEATURE_COLUMNS], test_df["label_enc"]

    print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows")
    print("Train label distribution:\n", train_df["label"].value_counts())
    print("Test label distribution:\n", test_df["label"].value_counts())

    # Inverse-frequency sample weights to combat extreme class imbalance
    class_counts = y_train.value_counts()
    weight_map = {cls: len(y_train) / (len(class_counts) * cnt) for cls, cnt in class_counts.items()}
    sample_weights = y_train.map(weight_map)

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="multi:softprob",
        num_class=len(le.classes_),
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train, sample_weight=sample_weights)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0, output_dict=True)
    print(classification_report(y_test, y_pred, target_names=le.classes_, zero_division=0))

    # --- Precision @ top-1% alert budget (realistic SOC analyst workload) ---
    # Score each test row by max anomaly probability (1 - P(normal))
    normal_idx = list(le.classes_).index("normal")
    anomaly_score = 1 - y_proba[:, normal_idx]
    top_k = max(1, int(0.01 * len(anomaly_score)))
    top_k_idx = np.argsort(-anomaly_score)[:top_k]
    true_is_anomaly = (y_test.values != normal_idx).astype(int)
    precision_at_top1pct = true_is_anomaly[top_k_idx].mean()
    print(f"\nPrecision @ top-1% alert budget ({top_k} alerts): {precision_at_top1pct:.3f}")
    print(f"Recall captured in top-1%: {true_is_anomaly[top_k_idx].sum()} / {true_is_anomaly.sum()} true anomalies")
    save_model(model, le)

    metrics_summary = {
        "classification_report": report,
        "precision_at_top1pct_alert_budget": float(precision_at_top1pct),
        "true_anomalies_in_test_set": int(true_is_anomaly.sum()),
        "anomalies_captured_in_top1pct": int(true_is_anomaly[top_k_idx].sum()),
        "train_rows": len(X_train),
        "test_rows": len(X_test),
        "classes": list(le.classes_),
    }
    with open(METRICS_OUT, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"\nSaved model, encoder, and metrics to {MODEL_OUT.rsplit('/',1)[0]}/")


if __name__ == "__main__":
    main()
