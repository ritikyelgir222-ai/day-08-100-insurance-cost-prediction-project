"""
Part of Phase 7: train/validation/test split
------------------------------------------------
WHY STRATIFY ON A CHARGES QUARTILE (not a plain random split): charges is
continuous, so there's no single class to stratify on. Binning charges
into quartiles and stratifying on that bin ensures train/val/test each
contain a similar mix of low-cost, mid-cost, and high-cost (largely
smoker-driven) individuals — the same reasoning used for the price-
quartile stratification in the house-price project, adapted here. This
matters more than usual in this specific dataset because the smoker
group is a small (~20%) but extremely high-cost minority — an unlucky
random split could easily over- or under-represent it in one of the
three sets.
"""

from sklearn.model_selection import train_test_split
import pandas as pd


def split_data(X, y, charges_for_stratify, test_size=0.15, val_size=0.15, random_state=42):
    charges_quartile = pd.qcut(charges_for_stratify, q=4, labels=False)

    X_temp, X_test, y_temp, y_test, q_temp, _ = train_test_split(
        X, y, charges_quartile, test_size=test_size, stratify=charges_quartile, random_state=random_state
    )
    val_relative_size = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_relative_size, stratify=q_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test
