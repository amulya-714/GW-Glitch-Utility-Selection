
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/metadata")

old_classes = [
    "1400Ripples", "1080Lines", "Air_Compressor", "Blip",
    "Chirp", "Extremely_Loud", "Helix", "Koi_Fish",
    "Light_Modulation", "Low_Frequency_Burst",
    "Low_Frequency_Lines", "No_Glitch", "None_of_the_Above",
    "Paired_Doves", "Power_Line", "Repeating_Blips",
    "Scattered_Light", "Scratchy", "Tomte", "Violin_Mode",
    "Wandering_Line", "Whistle"
]

for filename in ["H1_O2.csv", "L1_O2.csv",
                 "H1_O3a.csv", "L1_O3a.csv"]:

    df = pd.read_csv(DATA_DIR / filename)

    print(f"\n{'=' * 60}")
    print(filename)
    print("=" * 60)

    print("\nUnfiltered labels:")
    print(sorted(df["ml_label"].dropna().unique()))

    if "O3a" in filename:
        label_cols = [
            c for c in old_classes if c in df.columns
        ]

        predicted = df[label_cols].idxmax(axis=1)
        old_mask = df["ml_label"].isin(old_classes)
        new_mask = ~old_mask

        print("\nOld-class events:", old_mask.sum())
        print("Old-class mismatches:",
              (df.loc[old_mask, "ml_label"] !=
               predicted[old_mask]).sum())

        print("New/other-class events:", new_mask.sum())

    if filename == "L1_O3a.csv":
        duplicates = df[
            df.duplicated("gravityspy_id", keep=False)
        ]

        print("\nFull duplicate rows:")
        print(duplicates.to_string(index=False))
