#!/usr/bin/env python3
"""
SentinelLens -- run the whole pipeline with one command.

    python run_all.py

Executes every stage in the correct order and stops at the first failure with
a readable message, so there is no way to run the steps out of sequence.
Each stage can still be run individually (see SETUP.md); the individual
scripts are self-healing and will build any missing prerequisite features.
"""

import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))

STEPS = [
    ("Generate synthetic access logs",      "data/generate_synthetic_logs.py"),
    ("Engineer causal features",            "models/feature_engineering.py"),
    ("Graph-based detection model",         "models/graph_model.py"),
    ("Baseline profiling model",            "models/baseline_profiler.py"),
    ("Train detection model",               "models/train_model.py"),
    ("Explainability (SHAP rationales)",    "models/explainability.py"),
    ("Rule layer + streaming benchmark",    "models/hybrid_and_streaming.py"),
    ("Robustness (cold-start & drift)",     "models/robustness_eval.py"),
    ("Evaluation scorecard",                "models/evaluation_scorecard.py"),
]


def main():
    quiet = "--verbose" not in sys.argv
    print("=" * 64)
    print("SentinelLens - full pipeline".center(64))
    print("=" * 64)
    started = time.time()

    for i, (label, script) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {label}")
        print(f"        {script}")
        t0 = time.time()
        result = subprocess.run(
            [sys.executable, os.path.join(ROOT, script)],
            cwd=ROOT,
            capture_output=quiet,
            text=True,
        )
        if result.returncode != 0:
            print(f"\n  FAILED after {time.time()-t0:.1f}s\n")
            if quiet and result.stdout:
                print(result.stdout[-2000:])
            if quiet and result.stderr:
                print(result.stderr[-2000:])
            print("\nRe-run with --verbose for full output.")
            sys.exit(1)
        print(f"        done in {time.time()-t0:.1f}s")

    print("\n" + "=" * 64)
    print(f"Pipeline complete in {time.time()-started:.0f}s")
    print("=" * 64)
    print("\nResults written to reports/:")
    print("  model_metrics.json         per-class detection metrics")
    print("  hybrid_metrics.json        ML + rule-layer metrics")
    print("  robustness_metrics.json    cold-start & concept-drift false positives")
    print("  evaluation_scorecard.json  every judging criterion, measured")
    print("  alert_queue.csv            explained alerts")
    print("\nLaunch the analyst dashboard with:")
    print("  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
