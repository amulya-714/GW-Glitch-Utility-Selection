
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/metadata")

for filename in ["H1_O3a.csv", "L1_O3a.csv"]:
    df = pd.read_csv(DATA_DIR / filename)

    label_cols = [
        c for c in df.columns
        if c not in [
            "event_time", "ifo", "peak_time", "peak_time_ns",
            "start_time", "start_time_ns", "duration",
            "peak_frequency", "central_freq", "bandwidth",
            "channel", "amplitude", "snr", "q_value",
            "gravityspy_id", "ml_label", "ml_confidence",
            "url1", "url2", "url3", "url4"
        ]
    ]

    for col in label_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    print("\n", filename)
    print("Probability column types:")
    print(df[label_cols].dtypes.unique())

    predicted = df[label_cols].idxmax(axis=1)
    mismatch = df["ml_label"] != predicted

    sample = df.loc[mismatch].head(5)

    for _, row in sample.iterrows():
        print("\nID:", row["gravityspy_id"])
        print("ml_label:", row["ml_label"])
        print("Top probability labels:")
        print(row[label_cols].sort_values(ascending=False).head(5))
