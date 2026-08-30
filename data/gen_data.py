"""Generate a synthetic dataset of credit-invisible informal-economy workers.

We simulate ~10,000 users across three segments (street vendor, gig worker,
small farmer) with no formal credit history. Each user has several decades of
transactional/behavioral records that we later collapse into engineered
`features/build_features.py`. A hidden "true default risk" drives both the raw
records and a binary default label so the model has something learnable.

The point: these are the everyday signals a lender could reasonably read
beyond a bank statement. Everything is fake but statistically plausible.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def _segment_config(segment: str) -> tuple[float, float, float, float]:
    """Return (base_income, income_vol, savings_ratio, default_p) per segment."""
    if segment == "street_vendor":
        return 4200.0, 0.30, 0.12, 0.38
    if segment == "gig_worker":
        return 6800.0, 0.38, 0.18, 0.32
    return 5200.0, 0.55, 0.15, 0.42  # small_farmer


def generate_user(user_id: int, segment: str) -> dict:
    base_income, vol, base_save, base_def = _segment_config(segment)

    # Reliability / conscientiousness: the hidden latent trait we want to read.
    reliability = RNG.normal(0.0, 1.0)
    reliability = float(np.clip(reliability, -2.5, 2.5))

    # Sigmoid-ish link: higher reliability -> lower default probability.
    # Shifted so the population default rate sits near ~35-40% (realistic).
    odds = np.exp(-0.6 - 1.9 * reliability)
    default_p = odds / (1.0 + odds)
    default_p = 0.03 + 0.85 * default_p  # keep in (0.03, 0.88)
    is_default = RNG.random() < default_p

    account_months = int(RNG.integers(6, 72))
    kyc_complete = bool(RNG.random() < 0.7)
    digital_depth = int(np.clip(RNG.normal(2.5, 1.2), 0, 6))

    # Wallet/informal transaction history.
    n_paybacks = int(RNG.integers(1, 80))
    repay_scale = max(2.0, 15 - 12 * reliability)
    repayment_days = np.clip(RNG.normal(0, repay_scale), -30, 60)
    n_bills = int(RNG.integers(2, 60))
    bill_late_frac = float(np.clip(0.45 - 0.35 * reliability, 0.0, 0.9))

    # Income bursts drawn from a volatile stream.
    n_earnings = int(RNG.integers(4, 60))
    income_stream = np.abs(RNG.normal(base_income, base_income * vol, n_earnings))

    savings = max(0.0, base_save + 0.03 * reliability + RNG.normal(0, 0.05))
    savings = float(np.clip(savings, 0.0, 0.5))

    repeat_customers = float(np.clip(0.6 + 0.25 * reliability, 0.0, 1.0))
    on_time_fulfil = float(np.clip(0.85 + 0.11 * reliability, 0.0, 1.0))

    return {
        "user_id": f"U{user_id:05d}",
        "segment": segment,
        "reliability": reliability,
        "account_months": account_months,
        "kyc_complete": kyc_complete,
        "digital_depth": digital_depth,
        "income_mean": float(np.mean(income_stream)),
        "income_std": float(np.std(income_stream)),
        "n_income_bursts": n_earnings,
        "savings_ratio": savings,
        "repeat_customer_frac": repeat_customers,
        "on_time_fulfilment": on_time_fulfil,
        "n_informal_loans": n_paybacks,
        "repay_late_days": float(np.mean(repayment_days)),
        "bill_late_fraction": bill_late_frac,
        "n_bills_paid": n_bills,
        "is_default": is_default,
    }


def generate(n_users: int = 10_000) -> pd.DataFrame:
    segments = np.random.default_rng(0).choice(
        ["street_vendor", "gig_worker", "small_farmer"],
        size=n_users,
        p=[0.4, 0.35, 0.25],
    )
    rows = [generate_user(i, seg) for i, seg in enumerate(segments)]
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate()
    out = "data/dataset.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows to {out}")
    print(df["is_default"].value_counts(normalize=True).round(3))
