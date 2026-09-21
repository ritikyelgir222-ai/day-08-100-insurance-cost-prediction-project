# Project Documentation: Insurance Cost / Claim Severity Prediction
### Technical Documentation & Handover — Phase 12 of the SDLC

This document is the technical documentation and user manual a real handover
package would include. It complements `README.md` (setup and run commands)
by explaining the architecture, data, decisions, and results in enough
depth that someone who did not build this project could maintain or extend
it. Every number below came from actually running the code in this
repository — none are illustrative.

---

## 1. Project Summary

| | |
|---|---|
| **Objective** | Estimate expected annual cost per policyholder for risk-based pricing and reserve planning |
| **Client context** | Health insurer's actuarial / underwriting team |
| **Data source** | Medical Cost Personal Datasets — [Kaggle: mirichoi0218/insurance](https://www.kaggle.com/datasets/mirichoi0218/insurance) |
| **Dataset size** | 1,338 individuals (1,337 after deduplication), 6 raw features |
| **Final model** | XGBoost Regressor, 15 engineered features |
| **Test MAE** | $2,338.17 |
| **Business framing** | Price individual policyholders accurately enough to reduce aggregate reserving error, vs. a flat-rate-for-everyone approach |

---

## 2. Architecture

```
data/insurance.csv
        │
        ▼
data_loader.py ──────► loads raw CSV
        │
        ▼
clean_and_engineer.py ─► drops 1 duplicate row, encodes categoricals,
        │                engineers the smoker x BMI interaction
        ▼
   split.py ───────────► charges-quartile-stratified train/val/test split
        │
        ▼
train_model.py ───────► trains baseline (2-feature linear regression),
        │                candidate (random forest), and final model
        │                (XGBoost); evaluates; saves artifacts
        ▼
outputs/cost_model.joblib, feature_columns.joblib
        │
        ├──────────────► app.py ─── FastAPI service, /estimate endpoint,
        │                           reuses clean_and_engineer.py so
        │                           training and serving logic never
        │                           drift apart
        │
        └──────────────► monitor.py ─ PSI drift check + retrain trigger,
                                       compares against the baseline
                                       accuracy stored in business_validation.json
```

---

## 3. Data Dictionary (raw columns, as received)

| Column | Type | Description | Kept as-is / Transformed |
|---|---|---|---|
| `age` | int | Applicant age | Kept as-is |
| `sex` | string | male/female | Binary-encoded |
| `bmi` | float | Body mass index | Kept as-is, also binned into `bmi_category` |
| `children` | int | Number of dependents | Kept as-is |
| `smoker` | string | yes/no | Binary-encoded — the single strongest cost driver found in EDA |
| `region` | string | 4 US census regions | One-hot encoded — no ordinal relationship |
| `charges` | float | Actual annual medical insurance cost | **Target** — log1p-transformed for training (raw skew: 1.52) |

**Engineered features (not in the raw data):**

| Feature | Formula | Business rationale |
|---|---|---|
| `bmi_category` | WHO/CDC clinical bins (Underweight/Normal/Overweight/Obese) | Industry-standard thresholds, one-hot encoded |
| `smoker_bmi_interaction` | `smoker × bmi` | Captures the multiplicative (not additive) cost escalation EDA found between smoking and obesity |
| `is_smoker_and_obese` | `smoker == 1 AND bmi >= 30` | Explicit named flag for the ~5x-cost highest-risk segment |

Full reasoning for every decision above is inline in `clean_and_engineer.py`.

---

## 4. Key EDA Findings

From `eda.py`, run against the real dataset:

| Cut | Finding |
|---|---|
| Smoker status | Smokers average **$32,050**/year vs. **$8,434** for non-smokers — a **3.8x** difference, by far the single strongest driver |
| Target skew | Raw `charges` skew = **1.516** (right-skewed, driven by the smoker segment); log1p-transformed skew is much closer to normal |
| Region | Ranges from **$12,347** (southwest) to **$14,735** (southeast) — a modest effect compared to smoking |
| BMI category | Ranges from **$8,658** (Underweight) to **$15,561** (Obese) — a real but comparatively modest effect on its own |
| **Smoker × BMI interaction** | Non-smoking obese: **$8,853**. Smoking obese: **$41,693** — a **4.7x** jump, far larger than the independent smoker effect (~$23,600) and obesity effect (~$1,700) added together |
| Data quality | 1 exact duplicate row (every column, including `charges`, matched) |

The interaction finding is the centerpiece of this project's EDA — it's the direct justification for the `smoker_bmi_interaction` engineered feature, and it's the kind of non-obvious, checked-not-assumed finding this series is built around.

---

## 5. Modeling Results

### Experiment log (validation set)

| Model | Val MAE | Val % within ±20% |
|---|---|---|
| Linear Regression (baseline, age + smoker) | $5,133.63 | 56.72% |
| Random Forest | **$1,951.57** | 74.13% |
| **XGBoost (final)** | $2,212.86 | **74.63%** |

**Note on model selection:** Random Forest actually had a LOWER
validation MAE than XGBoost, but XGBoost was selected because it won on
`val_pct_within_20pct` — the metric tied to the actual business decision
(how many individual price estimates are close enough to act on). This
is the same kind of honestly-reported, metric-driven tradeoff used
throughout this series (see the attrition and sales-forecasting
projects' equivalent notes) — average error and "close enough often
enough" don't always agree, and the business-relevant one wins the tie.

### Held-out test set (final, unbiased evaluation)

| Metric | Value |
|---|---|
| Test MAE | $2,338.17 |
| Test MAPE | 19.35% |
| Test R² | 0.8584 |
| Test % within ±20% | 69.65% |
| N test individuals | 201 |

### What drives the model (feature importance)

Top 5 by importance:

1. `smoker` (0.4156)
2. `smoker_bmi_interaction` (0.3520)
3. `age` (0.0926)
4. `children` (0.0308)
5. `bmi` (0.0140)

`smoker` and `smoker_bmi_interaction` together account for **76.76%** of
the model's total decision weight — confirming both the raw EDA finding
and the value of engineering the interaction explicitly. Interestingly,
the binary `is_smoker_and_obese` flag ranked far lower (0.0076) than
its continuous counterpart — the model preferred the more granular
continuous interaction term over the binary summary of the same
underlying signal, a useful reminder that not every engineered feature
earns equal weight even when the underlying hypothesis is correct.

### Business validation — genuinely measured, not illustrative

On the 201-individual held-out test set:

| | |
|---|---|
| Naive flat-rate prediction (everyone priced at the training-set average) | $13,352.08 |
| Naive approach's total absolute reserving error | $1,826,446.39 |
| Model's total absolute reserving error | $469,971.31 |
| **Measured reserving-error reduction** | **$1,356,475.07 (74.27%)** |

Unlike every other project in this series, this figure required **no
externally-assumed rate or illustrative dollar amount** — `charges` is a
real dollar value in this dataset, so both the naive and model errors
are directly computed on the same real test cohort. This is the honest
highlight worth leading the LinkedIn post with: the METHOD (comparing
individualized predictions against a flat-rate baseline in real dollar
terms) is one every other project in this series had to approximate,
and this dataset is the one place it could be done exactly.

---

## 6. API Reference (Phase 10)

**Base URL (local):** `http://127.0.0.1:8000`

| Endpoint | Method | Purpose |
|---|---|---|
| `/` | GET | Browser-based test form (every field wired) |
| `/docs` | GET | Auto-generated interactive API docs (Swagger UI) |
| `/health` | GET | Health check, returns `{"status": "ok"}` |
| `/estimate` | POST | Estimate cost for a single applicant — see request/response shape below |

**Response (verified against two contrasting real applicants):**
```json
{
  "estimated_annual_cost": 46658.92,
  "top_factors": ["smoker", "smoker_bmi_interaction", "age"]
}
```
A young, healthy-BMI, non-smoking applicant scored **$3,791.36** on the
same model — confirming the API responds to genuinely different inputs,
not a fixed value.

---

## 7. Monitoring & Maintenance Plan (Phase 13)

`monitor.py` implements, and was run to confirm works correctly:

- **Feature drift check** via PSI on `age`, `bmi`, `smoker`,
  `smoker_bmi_interaction`. All four came back stable (PSI between
  0.002 and 0.056) — expected, since train/test here come from the same
  original snapshot.
- **Performance decay check**: compares current `% within ±20%` against
  the baseline; triggers a retrain recommendation if it drops more than
  0.05. On this run: no drop, so no retrain triggered.

**Important limitation stated directly in the script's output**: this
dataset has no timestamp, so genuine medical-cost-inflation drift (a
real and expected phenomenon in health insurance) cannot be evaluated
here — only that the monitoring code itself functions correctly. A
production deployment should also track whether predicted-vs-actual cost
LEVELS drift upward year over year, not just feature distributions.

---

## 8. Known Limitations (stated for the handover record)

1. **Small dataset** (1,337 rows post-deduplication) relative to what a
   real insurer's actuarial team would have access to.
2. **US-only, single snapshot in time** — no adjustment for medical cost
   inflation or more granular regional cost-of-care variation.
3. **The non-smoker majority (79.5% of rows) is comparatively less
   differentiated** by the model — most of its predictive power is
   concentrated in the smoker/BMI interaction, so individualized
   accuracy is stronger for the smoker minority than for the larger
   non-smoker group.
4. **No timestamp exists**, so genuine cost-inflation drift cannot be
   evaluated — see Section 7.
5. **No real claims/medical-history data** — only 6 demographic/lifestyle
   fields are available; a real underwriting model would incorporate
   medical history, prior claims, and more.

---

## 9. File Map (for quick reference)

| File | Phase | Purpose |
|---|---|---|
| `data_loader.py` | 5 | Load raw data, document source |
| `eda.py` | 6 | Smoker, BMI, region cuts, and the smoker x BMI interaction finding |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Deduplication, encoding, feature engineering — fully commented |
| `split.py` | 7 | Charges-quartile-stratified train/val/test split |
| `train_model.py` | 8-9 | Model training, evaluation, genuinely-measured business validation |
| `app.py` | 10 | FastAPI cost-estimation service + fully-wired test form |
| `monitor.py` | 13 | Drift detection, retrain trigger |
| `README.md` | 12 | Setup and run instructions |
| `PROJECT_DOCUMENTATION.md` (this file) | 12 | Technical documentation and handover |
