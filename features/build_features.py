"""Turn raw (simulated) transactional/behavioral records into model features.

Two feature sets are produced from the raw columns:

1. ALT features — the "beyond a bank statement" alternative signals
   (income stability, repayment punctuality, savings, digital footprint,
   community/seller reputation).

2. FORMAL features — a deliberately thin stand-in for what a bank could read
   without alternative data (mean income only + a couple of weak proxies).
   We use this as a BASELINE to prove the alternative signals add value.
"""

from __future__ import annotations

import pandas as pd
import numpy as np

ALT_FEATURES = [
    "account_months",
    "kyc_complete",
    "digital_depth",
    "income_volatility",
    "income_regularity",
    "n_income_bursts_log",
    "savings_ratio",
    "repeat_customer_frac",
    "on_time_fulfilment",
    "repay_punctuality",
    "repay_volume_log",
    "bill_reliability",
]

FORMAL_FEATURES = ["income_mean_log", "income_std_log"]


def build(raw: pd.DataFrame) -> pd.DataFrame:
    df = raw.copy()
    df = df.dropna(subset=["is_default"])
    df = df[df["income_mean"] > 0]

    eps = 1e-6

    # --- ALT features ---
    df["income_volatility"] = df["income_std"] / (df["income_mean"] + eps)
    # Regularity: how many distinct income bursts relative to account duration.
    df["income_regularity"] = df["n_income_bursts"] / (df["account_months"] + 1)
    df["n_income_bursts_log"] = np.log1p(df["n_income_bursts"])
    # Repayment: negative late-days = early, positive = late. Push to [0,1].
    df["repay_punctuality"] = 1.0 / (1.0 + np.exp(df["repay_late_days"] / 5.0))
    df["repay_volume_log"] = np.log1p(df["n_informal_loans"])
    df["bill_reliability"] = 1.0 - df["bill_late_fraction"]

    # --- FORMAL (thin baseline) features ---
    df["income_mean_log"] = np.log1p(df["income_mean"])
    df["income_std_log"] = np.log1p(df["income_std"])

    return df


def prepare_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    target = df["is_default"].astype(int)
    return df.drop(columns=["is_default"]), target


if __name__ == "__main__":
    from data.gen_data import generate

    raw = generate()
    feat = build(raw)
    print("ALT features:", ALT_FEATURES)
    print("FORMAL features:", FORMAL_FEATURES)
    print(f"rows={len(feat)} defaults={feat['is_default'].mean():.3f}")
    print(feat[ALT_FEATURES].describe().T.round(3))
