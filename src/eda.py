"""
Exploratory Data Analysis for the Water Potability dataset.

Generates:
  - reports/eda_summary.md   (text summary: shape, missingness, class balance, stats)
  - reports/*.png            (distribution plots, correlation heatmap, class balance)

Run:
    python src/eda.py
"""
import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "water_potability.csv")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

FEATURES = [
    "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
    "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
]


def main():
    os.makedirs(REPORTS_DIR, exist_ok=True)
    df = pd.read_csv(DATA_PATH)

    lines = []
    lines.append("# Water Potability — EDA Summary\n")
    lines.append(f"- Rows: **{df.shape[0]}**, Columns: **{df.shape[1]}**\n")

    # Missingness
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    lines.append("\n## Missing values\n")
    lines.append("| Column | Missing | % |\n|---|---|---|\n")
    for col in df.columns:
        if missing[col] > 0:
            lines.append(f"| {col} | {missing[col]} | {missing_pct[col]}% |\n")

    # Class balance
    counts = df["Potability"].value_counts()
    pct = (counts / len(df) * 100).round(1)
    lines.append("\n## Target balance (Potability)\n")
    lines.append(f"- Not potable (0): {counts.get(0, 0)} ({pct.get(0, 0)}%)\n")
    lines.append(f"- Potable (1): {counts.get(1, 0)} ({pct.get(1, 0)}%)\n")
    lines.append("\n**Note:** the dataset is moderately imbalanced (~61/39), so we use "
                  "`class_weight=\"balanced\"` and report precision/recall/F1 per class, "
                  "not just accuracy.\n")

    # Describe
    lines.append("\n## Summary statistics\n")
    lines.append(df[FEATURES].describe().T.round(2).to_markdown())
    lines.append("\n")

    with open(os.path.join(REPORTS_DIR, "eda_summary.md"), "w") as f:
        f.writelines(lines)

    # Plots
    sns.set_style("whitegrid")

    # 1. Class balance
    plt.figure(figsize=(5, 4))
    sns.countplot(x="Potability", data=df, palette=["#5c8ca8", "#e0793c"])
    plt.title("Target class balance")
    plt.xlabel("Potability (0 = not potable, 1 = potable)")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "class_balance.png"), dpi=120)
    plt.close()

    # 2. Missingness bar chart
    plt.figure(figsize=(7, 4))
    missing[missing > 0].sort_values().plot(kind="barh", color="#c9584a")
    plt.title("Missing values per column")
    plt.xlabel("Count missing")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "missing_values.png"), dpi=120)
    plt.close()

    # 3. Feature distributions by class
    fig, axes = plt.subplots(3, 3, figsize=(14, 10))
    for ax, col in zip(axes.flatten(), FEATURES):
        sns.kdeplot(data=df, x=col, hue="Potability", ax=ax, fill=True, common_norm=False,
                    palette=["#5c8ca8", "#e0793c"], legend=False)
        ax.set_title(col)
        ax.set_ylabel("")
    fig.suptitle("Feature distributions by potability class", y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "feature_distributions.png"), dpi=120, bbox_inches="tight")
    plt.close()

    # 4. Correlation heatmap
    plt.figure(figsize=(8, 6))
    corr = df.corr(numeric_only=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, square=True)
    plt.title("Feature correlation matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORTS_DIR, "correlation_heatmap.png"), dpi=120)
    plt.close()

    print("EDA complete.")
    print(f"- Summary: {os.path.join(REPORTS_DIR, 'eda_summary.md')}")
    print(f"- Plots saved to: {REPORTS_DIR}")


if __name__ == "__main__":
    main()
