import numpy as np
import pandas as pd


REQUIRED_COLUMNS = [
    "timestamp_id",
    "asset_id",
    "hour",
    "irradiance_wm2",
    "wind_speed_ms",
    "temperature_c",
    "vibration_mm_s",
    "power_mw",
]

NUMERIC_COLUMNS = [
    "timestamp_id",
    "hour",
    "irradiance_wm2",
    "wind_speed_ms",
    "temperature_c",
    "vibration_mm_s",
    "power_mw",
]


def validate_data(df):
    """
    Validate input data before preprocessing and feature engineering.
    Returns True if the basic validation checks pass.
    """

    # 1. Check whether the input is a DataFrame.
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input data must be a pandas DataFrame.")

    # 2. Check whether the dataset is empty.
    if df.empty:
        raise ValueError("Input dataset is empty.")

    # 3. Check required columns.
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # 4. Check missing asset identifiers.
    if df["asset_id"].isna().any():
        raise ValueError("asset_id contains missing values.")

    if df["asset_id"].astype(str).str.strip().eq("").any():
        raise ValueError("asset_id contains empty values.")

    if df["timestamp_id"].isna().any():
        raise ValueError("timestamp_id contains missing values.")

    # 5. Check numeric columns.
    for col in NUMERIC_COLUMNS:
        values = pd.to_numeric(df[col], errors="coerce")

        if values.isna().any():
            raise ValueError(
                f"{col} contains missing or non-numeric values."
            )

        if not np.isfinite(values).all():
            raise ValueError(
                f"{col} contains infinite values."
            )

    # 6. Check duplicate records.
    if df.duplicated().any():
        print("Warning: Exact duplicate rows detected.")

    # 7. Check duplicate timestamps for the same asset.
    if df.duplicated(
        subset=["asset_id", "timestamp_id"]
    ).any():
        print(
            "Warning: Duplicate asset_id and timestamp_id "
            "combinations detected."
        )

    # 8. Check valid hour values.
    if not df["hour"].between(0, 23).all():
        print("Warning: Some hour values are outside 0-23.")

    # 9. Report missing measurement values.
    measurement_columns = [
        "irradiance_wm2",
        "wind_speed_ms",
        "temperature_c",
        "vibration_mm_s",
        "power_mw",
    ]

    missing_measurements = df[measurement_columns].isna().sum()

    if missing_measurements.any():
        print("Warning: Missing measurement values detected.")
        print(missing_measurements[missing_measurements > 0])

    print("Data validation completed.")

    return True