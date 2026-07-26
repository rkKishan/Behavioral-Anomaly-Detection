"""
Version-safe model persistence.

Pickling an XGBoost model (joblib) ties the file to the exact XGBoost build
that created it, so loading it on another machine emits version warnings and
can break outright. XGBoost's native JSON format is portable across versions,
and the label encoder is stored as a plain class list, so no scikit-learn
internals are pickled either.
"""

import json
import os

import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

_MODELS = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = f"{_MODELS}/xgb_detector.json"
CLASSES_PATH = f"{_MODELS}/label_classes.json"


def save_model(model, label_encoder):
    model.save_model(MODEL_PATH)
    with open(CLASSES_PATH, "w") as f:
        json.dump(list(label_encoder.classes_), f, indent=1)


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(
            "\nERROR: trained model not found.\n"
            "Run:  python models/train_model.py\n")
    model = xgb.XGBClassifier()
    model.load_model(MODEL_PATH)
    with open(CLASSES_PATH) as f:
        classes = json.load(f)
    le = LabelEncoder()
    le.classes_ = np.array(classes)
    return model, le
