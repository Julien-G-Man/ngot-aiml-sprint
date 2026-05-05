# Cleans and splits the raw data into train and validation sets

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split


def preprocess(input_path: str, output_dir: str) -> None:
    """Clean data and create train/val splits."""
    print(f'Loading data from {input_path}...')
    df = pd.read_csv(input_path)
    print(f'Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns')

    # ── Feature Engineering ──────────────────────────────────
    # Rush hour flag
    rush = list(range(7, 10)) + list(range(17, 20))
    df['is_rush_hour'] = df['hour_of_day'].isin(rush).astype(float)

    # Log-transform weight (reduces skew from exponential distribution)
    df['log_weight'] = np.log1p(df['cargo_weight_kg'])

    # Traffic x Distance interaction feature
    df['effective_distance'] = df['distance_km'] * df['traffic_index']

    # One-hot vehicle type (if column exists)
    if 'vehicle_type' in df.columns:
        df['vehicle_truck']      = (df['vehicle_type'] == 'truck').astype(float)
        df['vehicle_van']        = (df['vehicle_type'] == 'van').astype(float)
        df['vehicle_motorcycle'] = (df['vehicle_type'] ==
'motorcycle').astype(float)
    else:
        df['vehicle_truck']      = 1.0
        df['vehicle_van']        = 0.0
        df['vehicle_motorcycle'] = 0.0

    # ── Remove Outliers ───────────────────────────────────────
    # Remove extreme ETAs (top 1%) — likely data entry errors
    q99 = df['eta_minutes'].quantile(0.99)
    n_before = len(df)
    df = df[df['eta_minutes'] <= q99].copy()
    print(f'Removed {n_before - len(df)} outlier rows (ETA > {q99:.1f} min)')

    # ── Train / Validation Split ──────────────────────────────
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

    # ── Save Output ───────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    train_df.to_csv(f'{output_dir}/train.csv', index=False)
    val_df.to_csv(f'{output_dir}/val.csv', index=False)

    print(f'Train set: {len(train_df)} rows  →  {output_dir}/train.csv')
    print(f'Val set:   {len(val_df)} rows   →  {output_dir}/val.csv')


if __name__ == '__main__':
    preprocess('data/raw/logistics_eta.csv', 'data/processed')
