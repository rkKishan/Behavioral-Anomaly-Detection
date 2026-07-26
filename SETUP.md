# SentinelLens — Setup & Run Guide

## Prerequisites
- Python 3.10+ (`python3 --version`)
- pip

## 1. Install
```
pip install -r requirements.txt
```

## 2. Run everything (recommended)
```
python3 run_all.py
```
Runs all nine stages in the correct order and stops at the first failure with
a readable message. Takes roughly 1–2 minutes. Add `--verbose` to see the full
output of each stage.

## 3. Launch the analyst dashboard
```
streamlit run dashboard/app.py
```
Opens at http://localhost:8501 — ranked alert queue, entity investigation,
model health, and a robustness tab (cold-start & concept drift).

---

## Running stages individually (optional)

The stages can also be run one at a time. They are **self-healing**: if a
prerequisite (such as the graph features) is missing, the script builds it
automatically rather than failing.

| Step | Command | What it produces |
|---|---|---|
| 1 | `python3 data/generate_synthetic_logs.py` | access_logs_labeled.csv (+unlabeled, +ground truth) |
| 2 | `python3 models/feature_engineering.py` | access_logs_featured.csv (18 causal features) |
| 3 | `python3 models/graph_model.py` | graph-based detector: entity↔resource features + AUC |
| 4 | `python3 models/baseline_profiler.py` | entity profiles + IsolationForest baseline |
| 5 | `python3 models/train_model.py` | trained model (xgb_detector.json) + model_metrics.json |
| 6 | `python3 models/explainability.py` | alert_queue.csv with SHAP rationales |
| 7 | `python3 models/hybrid_and_streaming.py` | rule-layer metrics + streaming benchmark |
| 8 | `python3 models/robustness_eval.py` | cold-start + concept-drift false positives |
| 9 | `python3 models/evaluation_scorecard.py` | evaluation_scorecard.json (every judging criterion) |

## Troubleshooting

**`python3: command not found`** — try `python` instead.

**`ModuleNotFoundError`** — the install step didn't complete:
`pip install -r requirements.txt` (or `pip3`).

**Numbers differ slightly from the report** — the generator draws fresh random
data each run, so metrics vary by a point or two. The committed files in
`reports/` correspond to the run described in `reports/README.md`. Run the
pipeline once before presenting and demo from those outputs.

See `reports/README.md` for architecture, assumptions, results, and limitations.
