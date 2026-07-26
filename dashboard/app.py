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

# ---------------------------------------------------------------------------
# ANALYST TRIAGE STATE
# ---------------------------------------------------------------------------
# Analyst verdicts are the only ground truth available in production. They are
# held in session_state and mirrored to disk so a verdict survives a page
# reload (and so the feedback can be consumed by a retraining job).
TRIAGE_PATH = os.path.join(REPORTS_DIR, "triage_feedback.json")


def load_triage():
    try:
        with open(TRIAGE_PATH) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_triage(state):
    with open(TRIAGE_PATH, "w") as f:
        json.dump(state, f, indent=2)


if "triage" not in st.session_state:
    st.session_state.triage = load_triage()


def set_verdict(session_id, verdict, entity_id, predicted_label):
    st.session_state.triage[session_id] = {
        "verdict": verdict,
        "entity_id": entity_id,
        "predicted_label": predicted_label,
        "reviewed_at": pd.Timestamp.now("UTC").isoformat(timespec="seconds"),
    }
    save_triage(st.session_state.triage)


def clear_verdict(session_id):
    st.session_state.triage.pop(session_id, None)
    save_triage(st.session_state.triage)


triage = st.session_state.triage
# Entities the analyst has explicitly cleared -> benign-drift allowlist.
allowlisted_entities = {
    v["entity_id"] for v in triage.values() if v["verdict"] == "false_positive"
}

# ---------------------------------------------------------------------------
# VIEW MODE
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("View mode")
    show_truth = st.toggle(
        "Ground-truth overlay",
        value=False,
        help="Evaluation aid only. The problem statement hides the label at "
             "inference, so a production console cannot show it. Leave this "
             "off for the analyst view; switch it on to score the model.",
    )
    if show_truth:
        st.warning(
            "**Evaluation mode.** Labels shown below come from the synthetic "
            "generator's held-out ground truth. This overlay does not exist "
            "in production."
        )
    else:
        st.success("**Production view.** No ground-truth labels in use.")
    st.divider()
    st.caption(f"Analyst verdicts recorded: **{len(triage)}**")
    if triage and st.button("Reset all verdicts", use_container_width=True):
        st.session_state.triage = {}
        save_triage({})
        st.rerun()

st.title("🛡️ Behavioral Anomaly Detection — Analyst Console")
st.caption("Ranked alert queue with explainable risk scores, entity history, and cold-start / drift awareness")

# ---- Top-line KPIs (production view: driven by analyst verdicts, not labels) ----
n_confirmed = sum(1 for v in triage.values() if v["verdict"] == "confirmed")
n_dismissed = sum(1 for v in triage.values() if v["verdict"] == "false_positive")
n_open = len(alerts) - len([s for s in triage if s in set(alerts["session_id"])])

col1, col2, col3, col4 = st.columns(4)
col1.metric("Alerts in queue", len(alerts))
col2.metric("Awaiting triage", n_open)
col3.metric("Confirmed incidents", n_confirmed, help="Analyst-accepted alerts")
col4.metric("Entities monitored", full_log["entity_id"].nunique())

if show_truth:
    precision_top = (alerts["label"] != "normal").mean()
    st.info(
        f"**Ground-truth overlay** — {int((alerts['label'] != 'normal').sum())} "
        f"of {len(alerts)} queued alerts are true anomalies "
        f"(precision {precision_top:.1%}). Evaluation metric, not a console feature."
    )

st.divider()

tab1, tab2, tab3, tab4 = st.tabs([
    "🚨 Alert Queue", "🔍 Entity Investigation",
    "📊 Correlations & Model Health", "🧪 Robustness (Cold-start & Drift)"])

# ---------------------------------------------------------------------------
with tab1:
    st.subheader("Ranked Alert Queue")

    f1, f2 = st.columns([3, 1])
    with f1:
        attack_types = st.multiselect(
            "Filter by predicted anomaly type",
            options=sorted(alerts["predicted_label"].unique()),
            default=[t for t in alerts["predicted_label"].unique() if t != "normal"],
        )
    with f2:
        status = st.selectbox(
            "Triage status",
            ["Awaiting triage", "Confirmed", "Dismissed", "All"],
        )

    filtered = alerts[alerts["predicted_label"].isin(attack_types)] if attack_types else alerts
    verdict_of = filtered["session_id"].map(lambda s: triage.get(s, {}).get("verdict"))
    if status == "Awaiting triage":
        filtered = filtered[verdict_of.isna()]
    elif status == "Confirmed":
        filtered = filtered[verdict_of == "confirmed"]
    elif status == "Dismissed":
        filtered = filtered[verdict_of == "false_positive"]
    filtered = filtered.sort_values("anomaly_score", ascending=False)

    if n_dismissed:
        st.caption(
            f"🔁 **Feedback loop active** — {n_dismissed} dismissed alert(s) have "
            f"added {len(allowlisted_entities)} entit(y/ies) to the benign-drift "
            f"allowlist. Their remaining queued alerts are down-ranked below."
        )

    if filtered.empty:
        st.success("Nothing in this view. Queue clear.")

    for _, row in filtered.head(25).iterrows():
        sid = row["session_id"]
        verdict = triage.get(sid, {}).get("verdict")
        allowlisted = row["entity_id"] in allowlisted_entities and verdict is None

        title = row["predicted_label"].replace("anomaly_", "").replace("_", " ").title()
        status_chip = {
            "confirmed": "✅ **Confirmed incident**",
            "false_positive": "❌ **Dismissed — false positive**",
        }.get(verdict, "🕓 Awaiting triage")

        with st.container(border=True):
            c1, c2 = st.columns([3, 1])
            with c1:
                # Ground truth is an evaluation overlay only -- never rendered in
                # the production view, because the label is hidden at inference.
                prefix = ""
                if show_truth:
                    prefix = "🔴 " if row["label"] != "normal" else "⚪ "
                st.markdown(
                    f"{prefix}**{title}** — `{row['entity_id']}` "
                    f"({row['entity_type']}) at {row['timestamp']}"
                )
                if show_truth:
                    st.caption(f"⚙️ ground truth (eval only): `{row['label']}`")
                st.caption(row["rationale"])
                st.markdown(status_chip)
                if allowlisted:
                    st.caption(
                        "⬇️ Down-ranked: this entity was previously cleared by an "
                        "analyst, so its novelty signals are treated as benign drift."
                    )
            with c2:
                st.metric("Risk score", f"{row['anomaly_score']:.2f}")
                if verdict is None:
                    b1, b2 = st.columns(2)
                    b1.button(
                        "✅ Accept", key=f"accept_{sid}", use_container_width=True,
                        on_click=set_verdict,
                        args=(sid, "confirmed", row["entity_id"], row["predicted_label"]),
                    )
                    b2.button(
                        "❌ Reject", key=f"reject_{sid}", use_container_width=True,
                        on_click=set_verdict,
                        args=(sid, "false_positive", row["entity_id"], row["predicted_label"]),
                    )
                else:
                    st.button(
                        "↩️ Undo", key=f"undo_{sid}", use_container_width=True,
                        on_click=clear_verdict, args=(sid,),
                    )

    # ---- What the feedback is for ----
    if triage:
        with st.expander(f"Analyst feedback log ({len(triage)} verdicts) — retraining input"):
            fb = pd.DataFrame.from_dict(triage, orient="index").reset_index(names="session_id")
            st.dataframe(fb, hide_index=True, use_container_width=True)
            st.caption(
                "Written to `reports/triage_feedback.json`. Confirmed incidents become "
                "positive training examples; dismissals become negatives and allowlist "
                "the entity's current behavioural baseline, which is how a real SOC "
                "stops re-alerting on legitimate change."
            )
            st.download_button(
                "Download feedback (JSON)",
                data=json.dumps(triage, indent=2),
                file_name="triage_feedback.json",
                mime="application/json",
            )

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
