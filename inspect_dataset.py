import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/metadata")

files = [
    "H1_O2.csv",
    "L1_O2.csv",
    "H1_O3a.csv",
    "L1_O3a.csv"
]

for filename in files:
    print("\n" + "=" * 60)
    print(f"FILE: {filename}")
    print("=" * 60)

    df = pd.read_csv(DATA_DIR / filename)

    print("Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string())

    print("\nMissing values:")
    print(df.isnull().sum().sort_values(ascending=False).head(10))

    print("\nData types:")
    print(df.dtypes)