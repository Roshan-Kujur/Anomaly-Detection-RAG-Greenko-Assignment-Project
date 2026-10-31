
import numpy as np
import pandas as pd


NUMERIC_COLUMNS = [
    "timestamp_id",
    "hour",
    "irradiance_wm2",
    "wind_speed_ms",
    "temperature_c",
    "vibration_mm_s",
    "power_mw",
]


def preprocess_data(df):
    """
    Clean and prepare the input dataset
    for feature engineering.
    """

    # 1. Preserve the original dataframe.
    data = df.copy()

    # 2. Remove whitespace from column names.
    data.columns = data.columns.str.strip()

    # 3. Convert numerical columns into numeric format.
    for col in NUMERIC_COLUMNS:
        if col in data.columns:
            data[col] = pd.to_numeric(
                data[col], errors="coerce"
            )

    # 4. Replace infinite values with NaN.
    data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True
    )

    # 5. Remove exact duplicate rows.
    data.drop_duplicates(inplace=True)

    # 6. Identify duplicate asset timestamps.
    duplicate_timestamps = data.duplicated(
        subset=["asset_id", "timestamp_id"],
        keep=False
    )

    if duplicate_timestamps.any():
        print(
            "Warning: Repeated asset_id and "
            "timestamp_id combinations found."
        )

    # 7. Sort data chronologically by asset.
    data.sort_values(
        ["asset_id", "timestamp_id"],
        inplace=True
    )

    data.reset_index(drop=True, inplace=True)

    print("Data preprocessing completed.")
    print("Final dataset shape:", data.shape)

    return data
