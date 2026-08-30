# 🔐 AltData Trust Score — credit for the credit-invisible

**Youth Economy Lab (YEL) IGDTUW — Hackathon prototype · Track 1 · Problem #1**
*Trust Scoring for the Credit-Invisible*

Millions of vendors, gig workers and small farmers are excluded from formal
credit — not because they're unreliable, but because banks can't read their
reliability without paperwork or a transaction history. This prototype turns
**everyday alternative signals** into a lender-trustworthy 300–850 score, and
explains *why* it was assigned so a lender can actually act on it.

## What it does
- Models default risk from **alternative data** (income rhythm, repayment
  punctuality on informal loans, savings, bill reliability, digital footprint,
  community/seller reputation) — not a bank statement.
- **Proves the value** by benchmarking against a thin "formal-data-only"
  baseline.
- Maps the model to a **300–850 trust score** and a **risk-based loan product**
  (approval, limit, rate tier), with the repayment structure tied to the
  segment's income pattern.
- **Explains each score with SHAP** contributions → a transparent, auditable
  decision.

## Results
| Model | Cross-validated AUC |
|-------|--------------------|
| Baseline — bank-only (formal) | **0.50** (≈ random) |
| Alternative-data model | **0.79** |

A **+0.28 AUC lift** — the exact story: banks can't see these users, but
alternative signals can. (Charts in `model/feature_imp.png`, `model/roc_alt.png`.)

## Project structure
```
yel-trust-score/
├── data/gen_data.py           # Synthetic 10k-user population (3 segments)
├── features/build_features.py # Raw records → ALT + FORMAL feature sets
├── model/train.py             # LightGBM vs baseline; saves model + metrics + charts
├── scoring/score.py           # proba → trust score → risk decision
├── scoring/explain.py         # SHAP explanation
├── app/demo.py                # Streamlit UI
└── README.md
```

## Run it
```bash
# 1. Dependencies
py -m pip install pandas numpy scikit-learn lightgbm shap streamlit matplotlib

# 2. (Re)build the dataset + model
py data\gen_data.py
py model\train.py

# 3. Launch the demo
py -m streamlit run app\demo.py
```

## What to demo
1. Click the **Street vendor / Gig worker** presets → **Prime**, approved, big
   limit.
2. Click **Small farmer** → **Sub-prime** — farm income swings + no KYC hurt the
   score, so the product restructures to **crop-cycle aligned repayments**.
3. Click **High-risk vendor** → **declined**, with a "path to credit" (build
   footprint, then re-score).
4. Show the **"Why this score?"** table and the **Baseline → Alternative AUC**
   comparison.
