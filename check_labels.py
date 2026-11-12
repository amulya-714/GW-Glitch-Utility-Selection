
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/metadata")

files = {
    "H1_O2": "H1_O2.csv",
    "L1_O2": "L1_O2.csv",
    "H1_O3a": "H1_O3a.csv",
    "L1_O3a": "L1_O3a.csv",
}

for name, filename in files.items():
    df = pd.read_csv(DATA_DIR / filename)

    print(f"\n{'=' * 50}")
    print(name, " | Total events:", len(df))
    print("=" * 50)

    print("\nLabel counts:")
    print(df["ml_label"].value_counts().to_string())

    print("\nUnique labels:", df["ml_label"].nunique())
    print("Duplicate gravityspy IDs:", df["gravityspy_id"].duplicated().sum())
    print("Missing spectrogram URLs:", df[["url1", "url2", "url3", "url4"]].isna().sum().sum())
