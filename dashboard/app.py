"""
SOC Analyst Dashboard - Behavioral Anomaly Detection
=======================================================
Run with: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import plotly.express as px
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "models"))

st.set_page_config(page_title="SOC Behavioral Anomaly Dashboard", layout="wide")

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "..", "reports")


@st.cache_data
def load_data():
    alerts = pd.read_csv(os.path.join(REPORTS_DIR, "alert_queue.csv"), parse_dates=["timestamp"])
    full_log = pd.read_csv(os.path.join(DATA_DIR, "access_logs_featured.csv"), parse_dates=["timestamp"])
    return alerts, full_log


alerts, full_log = load_data()

st.title("🛡️ Behavioral Anomaly Detection — Analyst Console")
st.caption("Ranked alert queue with explainable risk scores, entity history, and cold-start / drift awareness")

# ---- Top-line KPIs ----
col1, col2, col3, col4 = st.columns(4)
n_true_anomalies = (alerts["label"] != "normal").sum()
precision_top = (alerts["label"] != "normal").mean()
col1.metric("Alerts in queue", len(alerts))
col2.metric("True anomalies in top alerts", int(n_true_anomalies))
col3.metric("Precision (top alerts)", f"{precision_top:.0%}")
col4.metric("Entities monitored", full_log["entity_id"].nunique())

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Alert Queue", "🔍 Entity Investigation",
    "📊 Correlations & Model Health", "🧪 Robustness (Cold-start & Drift)"])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Ranked Alert Queue")
    attack_types = st.multiselect(
        "Filter by predicted anomaly type",
        options=sorted(alerts["predicted_label"].unique()),
        default=[t for t in alerts["predicted_label"].unique() if t != "normal"],
    )
    filtered = alerts[alerts["predicted_label"].isin(attack_types)] if attack_types else alerts
    filtered = filtered.sort_values("anomaly_score", ascending=False)

    for _, row in filtered.head(25).iterrows():
        is_true_positive = row["label"] != "normal"
        badge = "🔴" if is_true_positive else "⚪"
        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                st.markdown(
                    f"{badge} **{row['predicted_label'].replace('anomaly_', '').replace('_', ' ').title()}** "
                    f"— `{row['entity_id']}` ({row['entity_type']}) at {row['timestamp']}"
                )
                st.caption(row["rationale"])
            with c2:
                st.metric("Risk score", f"{row['anomaly_score']:.2f}")
                b1, b2 = st.columns(2)
                b1.button("✅ Accept", key=f"accept_{row['session_id']}")
                b2.button("❌ Reject", key=f"reject_{row['session_id']}")

# ---------------------------------------------------------------------------
with tab2:
    st.subheader("Entity History View")
    entity_id = st.selectbox("Select entity", sorted(full_log["entity_id"].unique()))
    ent_hist = full_log[full_log["entity_id"] == entity_id].sort_values("timestamp")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(f"**Total sessions:** {len(ent_hist)}")
        st.markdown(f"**Entity type:** {ent_hist['entity_type'].iloc[0]}")
        st.markdown(f"**Distinct resources accessed:** {ent_hist['resource_accessed'].nunique()}")
        st.markdown(f"**Distinct devices seen:** {ent_hist['device_fingerprint'].nunique()}")
        is_cold = ent_hist["is_cold_start_entity"].iloc[-1] == 1
        st.markdown(f"**Cold-start status:** {'⚠️ Yes — limited history, scored against population baseline' if is_cold else '✅ Established profile'}")

    with c2:
        fig = px.scatter(
            ent_hist, x="timestamp", y="hour_of_day", color="label",
            title="Session timing pattern (color = ground truth label)",
            color_discrete_map={"normal": "lightblue"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("**Resource access timeline**")
    fig2 = px.scatter(
        ent_hist, x="timestamp", y="resource_accessed", color="label",
        color_discrete_map={"normal": "lightblue"},
    )
    st.plotly_chart(fig2, use_container_width=True)

# ---------------------------------------------------------------------------
with tab3:
    st.subheader("Feature Correlations with Anomaly Outcomes")
    numeric_feats = [
        "geo_velocity_kmh", "time_since_last_session_min", "hour_deviation_zscore",
        "duration_zscore", "distinct_entities_per_ip_1h", "failed_auth_count_ip_1h",
        "failed_auth_count_entity_24h", "command_seq_length",
    ]
    full_log["is_anomaly"] = (full_log["label"] != "normal").astype(int)
    corr = full_log[numeric_feats + ["is_anomaly"]].corr()["is_anomaly"].drop("is_anomaly").sort_values()
    fig3 = px.bar(
        corr, orientation="h", title="Correlation of engineered features with anomaly label",
        labels={"value": "Correlation", "index": "Feature"},
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.subheader("Label Distribution (log scale)")
    label_counts = full_log["label"].value_counts()
    fig4 = px.bar(label_counts, log_y=True, title="Session counts by label (log scale — note extreme imbalance)")
    st.plotly_chart(fig4, use_container_width=True)

    st.info(
        "Model: XGBoost multiclass classifier, time-based train/test split (last 25% of days held out), "
        "inverse-frequency sample weighting for class imbalance. See reports/model_metrics.json for full "
        "classification report and precision-at-alert-budget figures."
    )

# ---------------------------------------------------------------------------
with tab4:
    st.subheader("Robustness: Cold-start Entities & Concept Drift")
    st.caption("A detector that flags every new employee or every shift change is "
               "unusable in production. These are measured, not assumed.")
    rpath = os.path.join(REPORTS_DIR, "robustness_metrics.json")
    if not os.path.exists(rpath):
        st.warning("Run `python models/robustness_eval.py` to generate these metrics.")
    else:
        with open(rpath) as f:
            rob = json.load(f)
        cs, cd = rob["cold_start"], rob["concept_drift"]

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### 🆕 Cold-start")
            st.metric("False-positive rate on brand-new entities",
                      f"{cs['false_positive_rate']:.2%}")
            st.caption(
                f"{cs['new_entities']} entities with **no prior history** "
                f"({cs['benign_sessions_from_new_entities']:,} benign sessions) produced "
                f"{cs['false_alerts']} false alerts. New entities are scored against a "
                "population-level profile for their entity type until they accrue history.")
        with c2:
            st.markdown("#### 📈 Concept drift")
            st.metric("False-positive rate after legitimate behaviour change",
                      f"{cd['false_positive_rate']:.2%}")
            st.caption(
                f"{cd['drifted_entities']} entities permanently changed shift pattern and "
                f"device ({cd['post_drift_benign_sessions']:,} sessions). Entity baselines are "
                "30-session rolling windows, so the new normal is re-learned instead of "
                "being flagged forever.")

        if "mean_hour_deviation_zscore_after_drift" in cd:
            st.markdown("**Evidence the profile adapted:**")
            comp = pd.DataFrame({
                "Cohort": ["Drifted entities (post-change)", "Baseline entities (no change)"],
                "Mean login-hour deviation (z)": [
                    cd["mean_hour_deviation_zscore_after_drift"],
                    cd["mean_hour_deviation_zscore_baseline_cohort"]],
            })
            st.dataframe(comp, hide_index=True, use_container_width=True)
            st.caption("Deviation for drifted entities is comparable to entities that never "
                       "changed — the rolling profile absorbed the new behaviour.")

        ins = rob.get("insider_drift_edge_case", {})
        if ins.get("sessions"):
            st.markdown("#### ⚖️ Insider drift (ambiguous edge case)")
            st.write(f"Flag rate: **{ins['flag_rate_at_1pct_budget']:.0%}** "
                     f"across {ins['sessions']} sessions")
            st.caption(ins.get("note", ""))
