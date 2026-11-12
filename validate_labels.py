
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/raw/metadata")

label_cols = [
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

    predicted = df[label_cols].idxmax(axis=1)
    mismatch = df["ml_label"] != predicted

    print(f"\n--- {filename} ---")
    print("Label/probability mismatches:", mismatch.sum())

    if mismatch.any():
        print(df.loc[mismatch, ["gravityspy_id", "ml_label"]].head(10))

    duplicates = df[df.duplicated("gravityspy_id", keep=False)]

    if not duplicates.empty:
        print("\nDuplicate IDs:")
        print(duplicates[
            ["gravityspy_id", "event_time", "peak_time", "ml_label"]
        ].to_string(index=False))
