
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/metadata")

files = [
    "H1_O2.csv",
    "L1_O2.csv",
    "H1_O3a.csv",
    "L1_O3a.csv"
]

thresholds = [0.7, 0.8, 0.9]

for filename in files:
    df = pd.read_csv(
        DATA_DIR / filename,
        usecols=["ml_label", "ml_confidence"]
    )

    print(f"\n{'=' * 65}")
    print(f"DATASET: {filename}")
    print("=" * 65)

    print("\nOverall confidence statistics:")
    print(df["ml_confidence"].describe().round(4))

    print("\nNumber of examples per class and confidence threshold:")

    counts = []

    for label, group in df.groupby("ml_label"):
        row = {
            "Class": label,
            "Total": len(group)
        }

        for threshold in thresholds:
            row[f">={threshold}"] = (
                group["ml_confidence"] >= threshold
            ).sum()

        counts.append(row)

    result = pd.DataFrame(counts)
    result = result.sort_values("Total", ascending=False)

    print(result.to_string(index=False))

    print("\nTotal retained at each threshold:")

    for threshold in thresholds:
        retained = (df["ml_confidence"] >= threshold).sum()
        percentage = retained / len(df) * 100

        print(
            f">={threshold}: {retained:,} "
            f"({percentage:.2f}%)"
        )
