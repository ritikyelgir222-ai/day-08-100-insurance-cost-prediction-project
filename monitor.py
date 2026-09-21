"""
Phase 13: Monitoring & Maintenance
--------------------------------------
WHY PSI (Population Stability Index) specifically: PSI is the standard,
interview-recognizable metric for feature drift in industry because it
gives a single interpretable number per feature with well-established
severity thresholds (< 0.1 stable, 0.1-0.25 moderate shift, > 0.25
significant shift needing action).

WHY THIS SCRIPT SIMULATES A "NEW BATCH" instead of using genuinely new
data: this dataset is a single snapshot with no natural second time
period to compare against. We reuse the held-out test set as a stand-in
"new batch" -- it's genuinely unseen by the model, so it's a reasonable
proxy for confirming the monitoring code path itself works, even though
in reality it's from the same original snapshot.
"""

import json

import numpy as np
import joblib

from data_loader import load_raw_data
from clean_and_engineer import clean_data, engineer_features, get_feature_columns
from split import split_data
from train_model import pct_within_tolerance

ACCURACY_DROP_THRESHOLD = 0.05  # per SDLC doc Phase 13: retrain if pct-within-20% drops more than this
DRIFT_FEATURES = ["age", "bmi", "smoker", "smoker_bmi_interaction"]


def population_stability_index(expected, actual, bins=10):
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    expected_pct = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_pct = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    expected_pct = np.clip(expected_pct, 1e-4, None)
    actual_pct = np.clip(actual_pct, 1e-4, None)
    return float(np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct)))


def run_monitoring_check():
    model = joblib.load("outputs/cost_model.joblib")
    feature_cols = joblib.load("outputs/feature_columns.joblib")

    raw = load_raw_data()
    cleaned = clean_data(raw)
    engineered = engineer_features(cleaned)

    X = engineered[feature_cols]
    charges = engineered["charges"]
    y = np.log1p(charges)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y, charges_for_stratify=charges)
    charges_test = np.expm1(y_test)

    print("=== Feature Drift (PSI): train distribution vs. held-out batch ===")
    print("PSI < 0.1: stable | 0.1-0.25: moderate | > 0.25: significant drift\n")
    for feat in DRIFT_FEATURES:
        psi = population_stability_index(X_train[feat], X_test[feat])
        flag = "SIGNIFICANT DRIFT" if psi > 0.25 else ("moderate" if psi > 0.1 else "ok")
        print(f"  {feat}: PSI={psi:.4f} [{flag}]")
    # WHY NO DRIFT IS EXPECTED HERE: train/test came from the same
    # stratified split of one snapshot, so we EXPECT near-zero PSI --
    # this run confirms the monitoring code itself works, not that real
    # demographic/cost drift is being caught. A live system would
    # replace X_test here with new applicant data collected after
    # deployment, and should additionally track whether overall cost
    # LEVELS are rising over time (medical cost inflation), which this
    # single-snapshot dataset can't demonstrate at all.

    pred_charges = np.expm1(model.predict(X_test))
    mape = float(np.mean(np.abs(pred_charges - charges_test) / charges_test))
    pct_within = pct_within_tolerance(charges_test, pred_charges)

    print(f"\n=== Performance on held-out batch ===")
    print(f"MAPE: {mape:.4f}")
    print(f"% within +/-20%: {pct_within:.4f}")

    with open("outputs/business_validation.json") as f:
        baseline_pct_within = json.load(f)["test_pct_within_20pct"]

    drop = baseline_pct_within - pct_within
    if drop > ACCURACY_DROP_THRESHOLD:
        print(f"\n⚠️  RETRAIN TRIGGERED: pct-within-20% dropped by {drop:.3f} (threshold: {ACCURACY_DROP_THRESHOLD})")
    else:
        print(f"\n✅ No retrain needed (pct-within-20% change: {drop:.3f}, threshold: {ACCURACY_DROP_THRESHOLD})")

    print("\nNOTE: a real deployment should also monitor whether average")
    print("predicted vs. actual cost LEVELS drift upward over time (medical")
    print("cost inflation), which a single annual snapshot cannot demonstrate.")


if __name__ == "__main__":
    run_monitoring_check()
