"""
Phases 5 (data quality fixes) & 7 (feature engineering)
------------------------------------------------------------
Every transformation and engineered feature below has a one-line WHY
comment next to it — the intent is that someone reviewing this file (a
stakeholder, a teammate, or future-you) can audit every decision without
having to guess at the reasoning.

FINAL FEATURE LIST is built at the bottom as FEATURE_COLUMNS.
"""

import pandas as pd

BINARY_COLUMNS = ["sex", "smoker"]
ONE_HOT_COLUMNS = ["region", "bmi_category"]


def clean_data(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    # -----------------------------------------------------------------
    # Fix 1: 1 exact duplicate row (verified in data_loader.py's output).
    # WHY DROP RATHER THAN INVESTIGATE FURTHER: with every column
    # (including the target `charges`) identical, this is almost
    # certainly a genuine duplicate record rather than two different
    # people who coincidentally match on every field including their
    # exact insurance charge to the cent — keeping it would silently
    # double-count one observation and slightly bias the model toward
    # whatever that one row's pattern happens to be.
    # -----------------------------------------------------------------
    df = df.drop_duplicates()

    return df


def engineer_features(clean_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_df.copy()

    # -----------------------------------------------------------------
    # Binary-encode sex and smoker
    # WHY: models need numeric input; both are true binary categories
    # with no ordinal meaning, so a simple 0/1 map is correct.
    # -----------------------------------------------------------------
    df["sex"] = (df["sex"] == "male").astype(int)
    df["smoker"] = (df["smoker"] == "yes").astype(int)

    # -----------------------------------------------------------------
    # BMI category (clinical standard bins)
    # BUSINESS LOGIC: these are the standard WHO/CDC BMI classification
    # thresholds used in actual clinical and actuarial underwriting
    # practice, not arbitrary bins chosen to fit this dataset — using
    # the industry-standard cutoffs means this feature would transfer
    # directly to a real underwriting system, not just this walkthrough.
    # -----------------------------------------------------------------
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )

    # -----------------------------------------------------------------
    # One-hot encode region and bmi_category — no ordinal relationship
    # between regions; bmi_category IS ordinal in principle, but EDA
    # found the smoker interaction (below) is what really matters, not a
    # smooth linear progression across categories, so one-hot (letting
    # the model learn each category's own effect and interaction
    # separately) is the safer choice over forcing an ordinal encoding.
    # -----------------------------------------------------------------
    df = pd.get_dummies(df, columns=ONE_HOT_COLUMNS, prefix=ONE_HOT_COLUMNS)
    dummy_cols = [c for c in df.columns if any(c.startswith(p + "_") for p in ONE_HOT_COLUMNS)]
    df[dummy_cols] = df[dummy_cols].astype(int)

    # -----------------------------------------------------------------
    # Engineered feature 1: smoker_bmi_interaction
    # BUSINESS LOGIC: this is the single most important finding from EDA
    # — smoking and high BMI don't just each add their own cost, they
    # COMPOUND. Non-smoking obese individuals average $8,853/year;
    # smoking obese individuals average $41,693/year — far more than
    # simply adding the independent smoker effect (~$23,600) and obesity
    # effect (~$1,700) together. A tree-based model CAN discover this
    # interaction on its own given enough data, but explicitly providing
    # it (smoker x bmi as a single continuous feature) makes the
    # relationship both easier for the model to use with fewer training
    # examples and directly explainable to an underwriter reviewing a
    # flagged high-cost case.
    # -----------------------------------------------------------------
    df["smoker_bmi_interaction"] = df["smoker"] * df["bmi"]

    # -----------------------------------------------------------------
    # Engineered feature 2: is_smoker_and_obese
    # BUSINESS LOGIC: a binary flag for the specific highest-risk segment
    # EDA identified (smoker AND bmi >= 30) — gives the model (and an
    # underwriter) an explicit, named category for the ~5x-cost segment,
    # on top of the continuous interaction term above.
    # -----------------------------------------------------------------
    df["is_smoker_and_obese"] = ((df["smoker"] == 1) & (df["bmi"] >= 30)).astype(int)

    return df


NUMERIC_MODEL_FEATURES = [
    "age", "sex", "bmi", "children", "smoker",
    "smoker_bmi_interaction", "is_smoker_and_obese",
]


def get_feature_columns(engineered_df: pd.DataFrame) -> list:
    dummy_cols = [c for c in engineered_df.columns if any(c.startswith(p + "_") for p in ONE_HOT_COLUMNS)]
    return NUMERIC_MODEL_FEATURES + dummy_cols


if __name__ == "__main__":
    from data_loader import load_raw_data

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)
    feature_cols = get_feature_columns(engineered)

    print(f"Raw rows: {len(raw)}  ->  Cleaned rows: {len(cleaned)} (after dropping duplicates)")
    print(f"Final feature columns ({len(feature_cols)}):")
    for c in feature_cols:
        print(f"  - {c}")
    print(f"\nis_smoker_and_obese rate: {engineered['is_smoker_and_obese'].mean():.3%}")

    engineered.to_csv("outputs/engineered_data.csv", index=False)
    print("\nSaved -> outputs/engineered_data.csv")
