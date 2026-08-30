"""Map the model's raw probability to a lender-facing 0-800 trust score
and attach a risk-based product decision (approval, limit, rate tier).
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
MODEL_PKL = os.path.join(ROOT, "model", "model.pkl")


def load_model():
    with open(MODEL_PKL, "rb") as f:
        return pickle.load(f)


def score_from_proba(proba: float) -> int:
    """Map default probability to a 300-850 trust score (higher = safer).

    Calibrated so the population median maps near ~600.
    """
    # Logistic-style mapping tuned on the simulated population.
    score = 610 + 240.0 * np.log((1.0 - proba) / (proba + 1e-9))
    return int(np.clip(score, 300, 850))


def risk_decision(score: int, segment: str) -> dict:
    """Return approval decision, loan limit cap, and interest tier."""
    if score >= 700:
        tier = "Prime"
        approved = True
        limit = 50000
        rate = 0.12
    elif score >= 580:
        tier = "Near-prime"
        approved = True
        limit = 20000
        rate = 0.18
    elif score >= 450:
        tier = "Sub-prime"
        approved = True
        limit = 8000
        rate = 0.26
    else:
        tier = "High-risk"
        approved = False
        limit = 0
        rate = None

    # Segment awareness: farmers get a slightly differeny product framing.
    note = {
        "street_vendor": "Working-capital micro-loan, daily repayment allowed",
        "gig_worker": "Income-flexible credit line, repay on burst days",
        "small_farmer": "Crop-cycle aligned credit, repay after harvest",
    }[segment]

    return {
        "approved": approved,
        "tier": tier,
        "limit": limit,
        "interest_rate": rate,
        "note": note,
    }


def raw_row_to_features(row: pd.DataFrame) -> pd.DataFrame:
    """Rebuild the ALT feature matrix from a raw user dict/row."""
    from features.build_features import build

    feat = build(row)
    return feat


def predict_proba_from_row(row: pd.DataFrame) -> float:
    from features.build_features import ALT_FEATURES

    model = load_model()["model"]
    feat = build(row)
    proba = float(model.predict_proba(feat[ALT_FEATURES])[:, 1][0])
    return proba


def predict_proba_from_features(features: pd.DataFrame, model=None) -> np.ndarray:
    if model is None:
        model = load_model()["model"]
    return model.predict_proba(features)[:, 1]
