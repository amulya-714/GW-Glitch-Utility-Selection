
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/metadata")
FILES = ["H1_O2.csv", "L1_O2.csv", "H1_O3a.csv", "L1_O3a.csv"]
THRESHOLD = 0.7

datasets = {}

for filename in FILES:
    df = pd.read_csv(
        DATA_DIR / filename,
        usecols=["ml_label", "ml_confidence"]
    )
    datasets[filename] = df[
        df["ml_confidence"] >= THRESHOLD
    ]

o2_classes = (
    set(datasets["H1_O2.csv"]["ml_label"])
    | set(datasets["L1_O2.csv"]["ml_label"])
)

o3_classes = (
    set(datasets["H1_O3a.csv"]["ml_label"])
    | set(datasets["L1_O3a.csv"]["ml_label"])
)

shared = sorted(o2_classes & o3_classes)
o2_only = sorted(o2_classes - o3_classes)
o3_only = sorted(o3_classes - o2_classes)

print("Shared classes:", len(shared))
print(shared)

print("\nO2-only classes:", o2_only)
print("\nO3a-only classes:", o3_only)

print("\nCLASS COUNTS AFTER CONFIDENCE FILTERING")
print("-" * 85)

rows = []

for label in shared:
    row = {"Class": label}

    for filename, df in datasets.items():
        row[filename.replace(".csv", "")] = (
            df["ml_label"] == label
        ).sum()

    rows.append(row)

result = pd.DataFrame(rows)
result["O2 total"] = result["H1_O2"] + result["L1_O2"]
result["O3a total"] = result["H1_O3a"] + result["L1_O3a"]

print(result.to_string(index=False))

print("\nTotal shared-class examples:")
print("O2:", result["O2 total"].sum())
print("O3a:", result["O3a total"].sum())
