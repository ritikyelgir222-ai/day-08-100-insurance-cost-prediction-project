# Insurance Cost / Claim Severity Prediction — Full Project Code (Real Dataset)

Companion code to Day 8 of the 100-day LinkedIn series. Uses the real
"Medical Cost Personal Datasets" (insurance.csv), run end to end — every
number in `PROJECT_DOCUMENTATION.md` came from executing this code, not
from an illustrative example.

## Data source

**Medical Cost Personal Datasets** (Kaggle: mirichoi0218/insurance) —
1,338 US individuals' age, sex, BMI, children, smoker status, region,
and their actual annual medical insurance charges.

The CSV is already included at `data/insurance.csv`. If you'd rather
pull it fresh from Kaggle: [the dataset
page](https://www.kaggle.com/datasets/mirichoi0218/insurance) — the
schema is identical.

## The headline finding: this project's business case needed no assumptions

Every other regression/classification project in this series had to
caveat its dollar business-impact figure with an externally-assumed rate
(a 35% retention-offer success rate, a $500 average fraud amount, a
1.5x-salary replacement cost). **This project doesn't** — `charges` is a
real dollar figure, so `train_model.py` computes a genuinely measured
comparison: what the total reserving error would be if the insurer
priced every policyholder at the flat historical average, versus what it
actually is using the model's individualized predictions, **on the same
held-out test cohort**. The result: a measured **$1,356,475 reduction
(74.3%)** in aggregate reserving error on 201 test individuals — a real
number, not an illustration of the method.

## Why every script is commented the way it is

- **The smoker x BMI interaction is the single most important finding**
  — non-smoking obese individuals average $8,853/year; smoking obese
  individuals average $41,693/year. That's far more than the smoker
  effect and obesity effect simply added together — a genuine
  multiplicative interaction, which directly motivated the
  `smoker_bmi_interaction` engineered feature (see
  `clean_and_engineer.py`).
- **BMI categories use the actual WHO/CDC clinical thresholds**, not
  arbitrary bins fit to this dataset — meaning the feature would
  transfer directly to a real underwriting system.
- **1 exact duplicate row is dropped, explicitly justified** — every
  column including the target matched exactly, which is not a
  coincidence worth preserving as two "different" observations.
- **Model selection is metric-driven, and honestly reported even when
  inconvenient** — Random Forest had a lower validation MAE than
  XGBoost (1,951.57 vs. 2,212.86), but XGBoost still won on the business
  metric (`% within ±20%`: 0.7463 vs. 0.7413) — the same kind of close,
  transparently-reported tradeoff made throughout this series.

## Setup

```bash
pip install -r requirements.txt
```

## Run order

```bash
python data_loader.py         # Phase 5 — loads & inspects the raw CSV
python eda.py                  # Phase 6 — smoker/BMI/region cuts + the interaction finding + outputs/eda_summary.png
python clean_and_engineer.py   # Phase 5 (fixes) + 7 — dedup, encoding, engineered features
python train_model.py          # Phase 8-9 — baseline + final model, measured business validation
python monitor.py              # Phase 13 — drift check + retrain-trigger simulation
```

## Serve the model (Phase 10)

```bash
uvicorn app:app --reload
```

```bash
curl -X POST http://127.0.0.1:8000/estimate \
  -H "Content-Type: application/json" \
  -d '{
        "age": 52, "sex": "male", "bmi": 34.5, "children": 1,
        "smoker": "yes", "region": "southeast"
      }'
```

Expected: an estimate around **$46,659** — this profile (older, smoker,
obese) matches the highest-risk segment EDA identified.

## File map

| File | SDLC Phase | What it does |
|---|---|---|
| `data_loader.py` | 5 | Loads the real dataset, documents its source |
| `eda.py` | 6 | Smoker, BMI category, region cuts, and the smoker x BMI interaction |
| `clean_and_engineer.py` | 5 (fixes) + 7 | Deduplication, encoding, and engineered interaction features — **fully commented with reasoning** |
| `split.py` | 7 | Charges-quartile-stratified train/val/test split |
| `train_model.py` | 8-9 | Baseline (2-feature linear regression) → Random Forest → XGBoost, measured business validation |
| `app.py` | 10 | FastAPI cost-estimation service — every form field wired, none hardcoded |
| `monitor.py` | 13 | PSI-based drift check + accuracy-drop retrain trigger |

## Outputs produced (in `outputs/`)

- `engineered_data.csv`
- `eda_summary.png`
- `experiment_log.csv` — baseline vs. candidate models
- `business_validation.json` — the genuinely measured reserving-error comparison
- `feature_importance.csv`
- `cost_model.joblib`, `feature_columns.joblib`
- `model_card.json` — includes explicit known limitations

## Known limitations (stated honestly, not hidden)

- **Small dataset** (1,337 rows after deduplication) — real premium/
  claims data would have far more rows and richer medical history.
- **US-only, single point in time** — no adjustment for medical cost
  inflation or regional cost-of-care differences beyond the 4 broad
  regions given.
- **The non-smoker segment (79.5% of the data) is comparatively less
  differentiated** — most of the model's predictive power comes from the
  smoker/BMI interaction, so performance on the much larger non-smoker
  majority is less sharply individualized.
- **No timestamp exists**, so `monitor.py` can't check for genuine cost
  drift over time (medical inflation) — only that the monitoring code
  runs correctly.
