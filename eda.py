"""
Phase 6: Exploratory Data Analysis
------------------------------------
WHY THESE SPECIFIC CUTS: an actuarial/underwriting stakeholder's first
question is "which factors actually drive cost, and do any of them
compound each other" — not a generic profile of every column. These cuts
test that directly, including checking whether smoking and obesity
interact rather than just each adding their own independent effect.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

from data_loader import load_raw_data


def run_eda(df: pd.DataFrame, out_dir: str = "outputs"):
    df = df.copy()

    print(f"Charges skew (raw): {df['charges'].skew():.3f}")
    print(f"Charges skew (log1p): {np.log1p(df['charges']).skew():.3f}")

    print("\n=== Average charges by smoker status ===")
    print(df.groupby("smoker")["charges"].agg(["mean", "median", "count"]).round(2))
    # WHY WE CHECK THIS FIRST: smoking is the single most commonly-cited
    # cost driver in health insurance underwriting — the natural first
    # sanity check before trusting anything else in the data.

    print("\n=== Correlation of numeric columns with charges ===")
    print(df[["age", "bmi", "children", "charges"]].corr()["charges"].round(3))

    print("\n=== Average charges by region ===")
    print(df.groupby("region")["charges"].mean().round(2))

    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 100],
        labels=["Underweight", "Normal", "Overweight", "Obese"],
    )
    print("\n=== Average charges by BMI category ===")
    print(df.groupby("bmi_category", observed=True)["charges"].mean().round(2))

    print("\n=== Average charges: smoker status x BMI category (the interaction) ===")
    interaction = df.groupby(["smoker", "bmi_category"], observed=True)["charges"].mean().round(2)
    print(interaction)
    # WHY THIS CUT MATTERS MOST: this is the single most important EDA
    # finding in this project. If smoking and obesity only had additive
    # effects, an obese smoker's cost would be roughly (obese effect) +
    # (smoker effect) above baseline. The actual jump from non-smoking
    # obese ($8,853) to smoking obese ($41,693) is far larger than that
    # — a genuine multiplicative interaction, not two independent
    # effects — which directly motivates the engineered interaction
    # feature in clean_and_engineer.py.

    print(f"\n=== Data quality ===")
    print(f"Missing values: {df.isnull().sum().sum()}")
    print(f"Duplicate rows: {df.duplicated().sum()} (handled in clean_and_engineer.py)")

    # ---- Chart: the three strongest cuts ----
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    sns.boxplot(data=df, x="smoker", y="charges", ax=axes[0], hue="smoker", legend=False, palette=["#55A868", "#C44E52"])
    axes[0].set_title("Charges by smoker status")

    df.groupby("bmi_category", observed=True)["charges"].mean().plot(
        kind="bar", ax=axes[1], color="#4C72B0", title="Avg charges by BMI category"
    )
    axes[1].tick_params(axis="x", rotation=20)

    interaction.unstack().plot(kind="bar", ax=axes[2], color=["#55A868", "#C44E52"], title="Smoker x BMI interaction")
    axes[2].tick_params(axis="x", rotation=20)
    axes[2].set_ylabel("Avg charges")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/eda_summary.png", dpi=120)
    print(f"\nSaved chart -> {out_dir}/eda_summary.png")


if __name__ == "__main__":
    data = load_raw_data()
    run_eda(data)
