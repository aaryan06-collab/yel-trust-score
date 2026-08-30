"""Streamlit demo -- Alternative Data Trust Score for the Credit-Invisible.

Quick start (from the project root):
    py -m streamlit run app/demo.py

The left panel edits everyday signals; the model scores the vendor and
explains the decision with SHAP-style contribution bars.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import streamlit as st

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from features.build_features import build, ALT_FEATURES  # noqa: E402
from scoring.score import (  # noqa: E402
    load_model,
    score_from_proba,
    risk_decision,
)
from scoring.explain import explain_row  # noqa: E402

MODEL_FOUND = os.path.exists(os.path.join(ROOT, "model", "model.pkl"))

st.set_page_config(page_title="AltData Trust Score", page_icon="🔐", layout="wide")

# ---------------------------------------------------------------------------
# Presets per segment -- plausible everyday signals for a credit-invisible user
# ---------------------------------------------------------------------------
PRESETS = {
    "Street vendor (daily cash, no bank)": {
        "segment": "street_vendor",
        "account_months": 30, "kyc_complete": True, "digital_depth": 3,
        "income_mean": 4300, "income_std": 1400, "n_income_bursts": 26,
        "savings_ratio": 0.14, "repeat_customer_frac": 0.82,
        "on_time_fulfilment": 0.93, "n_informal_loans": 12,
        "repay_late_days": -2.0, "bill_late_fraction": 0.06, "n_bills_paid": 18,
    },
    "Gig worker (ride/food, bursty pay)": {
        "segment": "gig_worker",
        "account_months": 14, "kyc_complete": True, "digital_depth": 5,
        "income_mean": 7100, "income_std": 3600, "n_income_bursts": 42,
        "savings_ratio": 0.20, "repeat_customer_frac": 0.71,
        "on_time_fulfilment": 0.88, "n_informal_loans": 7,
        "repay_late_days": 1.0, "bill_late_fraction": 0.14, "n_bills_paid": 9,
    },
    "Small farmer (crop-cycle income)": {
        "segment": "small_farmer",
        "account_months": 48, "kyc_complete": False, "digital_depth": 1,
        "income_mean": 5200, "income_std": 4100, "n_income_bursts": 6,
        "savings_ratio": 0.06, "repeat_customer_frac": 0.55,
        "on_time_fulfilment": 0.7, "n_informal_loans": 20,
        "repay_late_days": 4.0, "bill_late_fraction": 0.4, "n_bills_paid": 5,
    },
    "High-risk vendor (thin footprint)": {
        "segment": "street_vendor",
        "account_months": 4, "kyc_complete": False, "digital_depth": 0,
        "income_mean": 1600, "income_std": 1300, "n_income_bursts": 3,
        "savings_ratio": 0.01, "repeat_customer_frac": 0.3,
        "on_time_fulfilment": 0.45, "n_informal_loans": 1,
        "repay_late_days": 14.0, "bill_late_fraction": 0.7, "n_bills_paid": 1,
    },
}


def row_from(d: dict) -> pd.DataFrame:
    d = dict(d)
    d["user_id"] = "demo"
    d["is_default"] = False  # dropped downstream; placeholder
    return pd.DataFrame([d])


def build_sidebar() -> pd.DataFrame:
    with st.sidebar:
        st.markdown("## 👤 Vendor profile")
        preset = st.selectbox(
            "Load a preset",
            ["--- custom ---"] + list(PRESETS.keys()),
        )
        base = PRESETS.get(preset, PRESETS["Street vendor (daily cash, no bank)"])

        segment = st.selectbox(
            "Occupational segment",
            ["street_vendor", "gig_worker", "small_farmer"],
            index=["street_vendor", "gig_worker", "small_farmer"].index(
                base["segment"]
            ),
        )

        st.markdown("**Identity & footprint**")
        account_months = st.slider("Account age (months)", 1, 72, int(base["account_months"]))
        kyc_complete = st.checkbox("Completed KYC", value=bool(base["kyc_complete"]))
        digital_depth = st.slider("Digital footprint depth (0-6)", 0, 6, int(base["digital_depth"]))

        st.markdown("**Income rhythm**")
        income_mean = st.slider("Avg monthly income (₹)", 500, 15000, int(base["income_mean"]), 100)
        income_std = st.slider("Income month-to-month swing (₹)", 100, 8000, int(base["income_std"]), 100)
        n_income_bursts = st.slider("Income events / year", 1, 120, int(base["n_income_bursts"]))
        savings_ratio = st.slider("Savings as % of income", 0.0, 0.5, float(base["savings_ratio"]), 0.01)

        st.markdown("**Repayment & reliability**")
        n_informal_loans = st.slider("Prior informal loans repaid", 0, 80, int(base["n_informal_loans"]))
        repay_late_days = st.slider(
            "Avg repayment delay (days, − = early)", -30, 60,
            int(base["repay_late_days"]), 1,
        )
        bill_late_fraction = st.slider("Share of bills paid late", 0.0, 0.9, float(base["bill_late_fraction"]), 0.01)
        n_bills_paid = st.slider("Recurring bills paid", 0, 60, int(base["n_bills_paid"]))

        st.markdown("**Community / seller reputation**")
        repeat_customer_frac = st.slider("Repeat-customer ratio", 0.0, 1.0, float(base["repeat_customer_frac"]), 0.01)
        on_time_fulfilment = st.slider("On-time fulfilment/NA (0-1)", 0.0, 1.0, float(base["on_time_fulfilment"]), 0.01)

    return row_from({
        "segment": segment,
        "account_months": account_months,
        "kyc_complete": kyc_complete,
        "digital_depth": digital_depth,
        "income_mean": income_mean,
        "income_std": income_std,
        "n_income_bursts": n_income_bursts,
        "savings_ratio": savings_ratio,
        "repeat_customer_frac": repeat_customer_frac,
        "on_time_fulfilment": on_time_fulfilment,
        "n_informal_loans": n_informal_loans,
        "repay_late_days": repay_late_days,
        "bill_late_fraction": bill_late_fraction,
        "n_bills_paid": n_bills_paid,
    })


def fmt_contribution(c: float) -> str:
    sign = "+" if c >= 0 else ""
    return f"{sign}{c:+.3f}"


def main():
    if not MODEL_FOUND:
        st.error("Model not found. Run `py model\\train.py` from the project root first.")
        st.stop()

    st.markdown("## 🔐 AltData Trust Score — credit for the credit-invisible")
    st.caption(
        "Millions of vendors, gig workers and small farmers are excluded from credit "
        "because banks can't read reliability from a standard bank statement. "
        "Adjust everyday signals on the left — the model predicts default, scores the "
        "user 300–850, and explains *why*."
    )

    raw = build_sidebar()

    if raw.iloc[0]["income_mean"] <= 0:
        st.warning("Income must be positive.")
        st.stop()

    features = build(raw)
    model = load_model()["model"]
    proba = float(model.predict_proba(features[ALT_FEATURES])[:, 1][0])
    score = score_from_proba(proba)

    segment = raw.iloc[0]["segment"]
    decision = risk_decision(score, segment)

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        st.metric("Trust score", f"{score}", delta=None)
        st.caption("300–850 · higher = safer")
    with c2:
        st.metric("Predicted default prob.", f"{proba*100:.1f}%")
    with c3:
        st.metric("Decision", "✅ Approved" if decision["approved"] else "❌ Declined")

    st.divider()

    r1, r2 = st.columns([1, 1.1])
    with r1:
        st.markdown(f"### 📋 Risk product ({decision['tier']})")
        if decision["approved"]:
            st.write(f"- **Loan limit:** ₹{decision['limit']:,.0f}")
            st.write(f"- **Interest rate:** {decision['interest_rate']*100:.0f}%/yr")
            st.write(f"- **Structure:** {decision['note']}")
        else:
            st.write("- **Status:** pre-approval not granted")
            st.write("- **Path forward:** build footprint (KYC, bills, savings), then re-score")

    with r2:
        st.markdown("### 🧠 Why this score?")
        contributions, base = explain_row(features)
        # Only show the biggest drivers.
        top = contributions[:7]
        df_exp = pd.DataFrame({
            "Feature": [t["feature"] for t in top],
            "Contribution": [t["contribution"] for t in top],
            "Effect": [
                "Raises default risk" if t["contribution"] > 0 else "Lowers default risk"
                for t in top
            ],
        })
        st.dataframe(df_exp, use_container_width=True, hide_index=True)
        st.caption(
            "Positive contribution = pushes default probability up (hurts score); "
            "negative = helps. Sum of contributions + intercept = log-odds."
        )

    st.divider()
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### Feature importance (model)")
        imp_path = os.path.join(ROOT, "model", "feature_imp.png")
        if os.path.exists(imp_path):
            st.image(imp_path)
    with col_b:
        st.markdown("#### ROC — alternative data model")
        roc_path = os.path.join(ROOT, "model", "roc_alt.png")
        if os.path.exists(roc_path):
            st.image(roc_path)

    metrics_path = os.path.join(ROOT, "model", "metrics.json")
    if os.path.exists(metrics_path):
        import json

        with open(metrics_path) as f:
            m = json.load(f)
        st.info(
            f"Baseline (bank-only) CV AUC **{m['baseline_formal_cv_auc']:.2f}** → "
            f"Alternative-data CV AUC **{m['alternative_cv_auc']:.2f}** "
            f"(**, {m['auc_lift_vs_baseline']:.2f} lift**). "
            f"That gap is exactly the story: banks can't see these users; "
            f"alternative signals can."
        )


if __name__ == "__main__":
    main()
