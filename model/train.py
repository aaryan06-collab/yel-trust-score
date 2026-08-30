"""Train the alternative-data trust model and a thin formal-data baseline.

Demonstrates that alternative signals meaningfully beat a bank's formal-only
view when it comes to predicting default among credit-invisible users. Saves:

    model/model.pkl        - trained LightGBM
    model/metrics.json     - cross-validated AUC comparison
    model/feature_imp.png  - gain-based feature importance (top 15)
    model/roc_alt.png      - ROC curve for the alternative-data model
"""

from __future__ import annotations

import json
import os
import pickle
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
import lightgbm as lgb

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
DATA_CSV = os.path.join(ROOT, "data", "dataset.csv")
MODEL_PKL = os.path.join(ROOT, "model", "model.pkl")
METRICS_JSON = os.path.join(ROOT, "model", "metrics.json")
IMP_PNG = os.path.join(ROOT, "model", "feature_imp.png")
ROC_PNG = os.path.join(ROOT, "model", "roc_alt.png")

LGB_PARAMS = {
    "objective": "binary",
    "n_estimators": 400,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "verbosity": -1,
}


def load_features() -> pd.DataFrame:
    from data.gen_data import generate
    from features.build_features import build

    raw = generate()
    return build(raw)


def plot_importance(model, cols, path):
    imp = pd.Series(model.feature_importances_, index=cols).sort_values(
        ascending=False
    ).head(15)
    imp.sort_values().plot.barh(figsize=(7, 6), color="#2f6f4f")
    plt.title("Top 15 Features (gain)")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def plot_roc(y_true, proba, path):
    fpr, tpr, _ = roc_curve(y_true, proba)
    auc = roc_auc_score(y_true, proba)
    plt.plot(fpr, tpr, lw=2, label=f"ALT model (AUC = {auc:.3f})")
    plt.plot([0, 1], [0, 1], "--", color="grey")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC — Alternative Data Model")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    return auc


def main():
    from features.build_features import (
        ALT_FEATURES,
        FORMAL_FEATURES,
        prepare_target,
    )

    feat = load_features()
    X, y = prepare_target(feat)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    # --- Baseline: formal-only ---
    base_clf = lgb.LGBMClassifier(**LGB_PARAMS)
    base_cv = cross_val_score(
        base_clf, X[FORMAL_FEATURES], y, cv=5, scoring="roc_auc"
    ).mean()

    # --- ALT model ---
    clf = lgb.LGBMClassifier(**LGB_PARAMS)
    alt_cv = cross_val_score(clf, X[ALT_FEATURES], y, cv=5, scoring="roc_auc").mean()

    clf.fit(X_train[ALT_FEATURES], y_train)
    y_proba = clf.predict_proba(X_test[ALT_FEATURES])[:, 1]
    test_auc = roc_auc_score(y_test, y_proba)
    report = classification_report(
        y_test, (y_proba >= 0.5).astype(int), output_dict=True
    )

    plot_importance(clf, ALT_FEATURES, IMP_PNG)
    alt_roc = plot_roc(y_test, y_proba, ROC_PNG)

    metrics = {
        "baseline_formal_cv_auc": round(float(base_cv), 4),
        "alternative_cv_auc": round(float(alt_cv), 4),
        "alternative_test_auc": round(float(test_auc), 4),
        "auc_lift_vs_baseline": round(float(alt_cv - base_cv), 4),
        "default_rate": float(y.mean()),
        "classification_report": {
            k: {kk: float(vv) if isinstance(vv, float) else vv for kk, vv in v.items()}
            if isinstance(v, dict)
            else float(v)
            for k, v in report.items()
        },
    }

    os.makedirs(os.path.dirname(MODEL_PKL), exist_ok=True)
    with open(MODEL_PKL, "wb") as f:
        pickle.dump({"model": clf, "features": ALT_FEATURES}, f)
    with open(METRICS_JSON, "w") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps(metrics, indent=2))
    print("Saved model.pkl, metrics.json, feature_imp.png, roc_alt.png")


if __name__ == "__main__":
    main()
