"""
Phase 5: Data Collection & Data Understanding
------------------------------------------------
DATA SOURCE
-----------
This project uses the "Medical Cost Personal Datasets" (often just called
"insurance.csv"), a well-known benchmark dataset of 1,338 US individuals'
demographic/lifestyle attributes and their actual annual medical insurance
charges:
    https://www.kaggle.com/datasets/mirichoi0218/insurance

If you're following along on Kaggle: download `insurance.csv` from the
link above and place it at `data/insurance.csv` — the schema is
identical to the file already included in this project.

WHY THIS DATASET
-----------------
- `charges` is a REAL dollar figure (not an anonymized or synthetic
  proxy), which matters a lot for this project specifically: it lets the
  business-validation step in Phase 9 compute a genuinely measured
  reserving-error figure directly from the data, rather than needing an
  externally-assumed dollar conversion the way the fraud-detection and
  sales-forecasting projects in this series had to.
- Its 6 features (age, sex, BMI, children, smoker status, region) are
  fully interpretable, which supports the kind of business-perspective
  EDA this series is built around — and, as EDA below shows, they
  combine in a genuinely interesting, non-additive way (smoking and
  obesity interact multiplicatively on cost, not just additively).
- WHY THIS DATASET FITS THE "CLAIM SEVERITY" FRAMING: `charges` here is
  each individual's actual annual healthcare cost — functionally the
  same prediction problem an insurer's actuarial team faces when
  estimating expected cost (severity) per policyholder for risk-based
  pricing and reserve planning, even though the column is literally
  named "charges" rather than "claim amount."
"""

import pandas as pd

RAW_DATA_PATH = "data/insurance.csv"


def load_raw_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


if __name__ == "__main__":
    df = load_raw_data()
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns from {RAW_DATA_PATH}")
    print(f"\nCharges summary:")
    print(df["charges"].describe())
    print(f"\nCharges skew: {df['charges'].skew():.3f} (right-skewed — see clean_and_engineer.py)")
    print(f"\nMissing values: {df.isnull().sum().sum()}  |  Duplicate rows: {df.duplicated().sum()}")
