"""Explain a trust score with SHAP so a lender can see *why* it was assigned.

The model is a tree ensemble, so TreeExplainer is fast and exact. We return the
per-feature SHAP contributions for a single user, sorted by absolute impact.
"""

from __future__ import annotations

import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL_PKL = os.path.join(ROOT, "model", "model.pkl")


def build_explainer():
    with open(MODEL_PKL, "rb") as f:
        artifact = pickle.load(f)
    model = artifact["model"]
    try:
        import shap

        explainer = shap.TreeExplainer(model)
        return explainer
    except Exception:
        # Fall back to per-feature permutation-like contribution using raw
        # tree split info is complex; degrade to feature importances.
        return None


def explain_row(features: pd.DataFrame, model=None) -> list[dict]:
    """Return list of {feature, value, contribution} sorted by |contribution|."""
    import shap

    if model is None:
        with open(MODEL_PKL, "rb") as f:
            artifact = pickle.load(f)
        model = artifact["model"]
        features = features[artifact["features"]]

    explainer = shap.TreeExplainer(model)
    base = float(explainer.expected_value)
    sv = explainer.shap_values(features)

    # LightGBM binary classifiers may return a list of two SHAP matrices
    # (one per class); pick the positive-class matrix deterministically.
    if isinstance(sv, list):
        sv = sv[1] if len(sv) > 1 else sv[0]
    sv = np.asarray(sv)[0]  # single user -> 1D array of contributions

    out = []
    for col, val, contrib in zip(features.columns, features.iloc[0], sv):
        out.append({"feature": col, "value": float(val), "contribution": float(contrib)})
    out.sort(key=lambda d: abs(d["contribution"]), reverse=True)
    return out, base
