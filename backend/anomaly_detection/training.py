

# Training pipeline using Rolling Median/MAD algorithm

import json
from pathlib import Path

import numpy as np
import pandas as pd


# Detector configuration
WINDOW_SIZE = 6
MIN_PERIODS = 3
ANOMALY_THRESHOLD = 3.5

ASSET_FEATURES = {
    "solar": [
        "power_mw",
        "vibration_mm_s",
        "irradiance_wm2",
    ],
    "wind": [
        "power_mw",
        "vibration_mm_s",
        "wind_speed_ms",
    ],
}

REQUIRED_COLUMNS = [
    "timestamp_id",
    "asset_id",
    "asset_type",
]

def train_detector(training_df, output_dir=None):

    """
    Calibrate the Rolling Median/MAD detector using
    historical training data and save its configuration.

    The input dataframe is not modified.
    """

    # 1. Validate required columns
    missing_columns = [
        col for col in REQUIRED_COLUMNS
        if col not in training_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if training_df.empty:
        raise ValueError("Training dataframe is empty.")

    if training_df[
        ["asset_id", "timestamp_id", "asset_type"]
    ].isna().any().any():

        raise ValueError(
            "Asset ID, timestamp ID and asset type "
            "cannot contain missing values."
        )

    if training_df.duplicated(
        subset=["asset_id", "timestamp_id"]
    ).any():

        raise ValueError(
            "Duplicate asset_id and timestamp_id combinations found."
        )

    # 2. Prepare data
    data = training_df.copy()

    data["asset_type"] = (
        data["asset_type"].astype(str).str.strip().str.lower()
    )

    unsupported = (
        set(data["asset_type"]) - set(ASSET_FEATURES)
    )

    if unsupported:
        raise ValueError(
            f"Unsupported asset types: {sorted(unsupported)}"
        )

    # Validate timestamp ordering values
    data["timestamp_id"] = pd.to_numeric(
        data["timestamp_id"], errors="coerce"
    )

    if data["timestamp_id"].isna().any():
        raise ValueError(
            "timestamp_id must contain valid numeric values."
        )

    data = data.sort_values(
        ["asset_id", "timestamp_id"]
    )

    # 3. Validate required sensor features
    all_features = set(
        feature
        for features in ASSET_FEATURES.values()
        for feature in features
    )

    missing_features = [
        feature for feature in all_features
        if feature not in data.columns
    ]

    if missing_features:
        raise ValueError(
            f"Missing sensor features: {missing_features}"
        )

    # 4. Prepare configuration
    configuration = {
        "method": "Rolling Median/MAD",
        "window_size": WINDOW_SIZE,
        "min_periods": MIN_PERIODS,
        "anomaly_threshold": ANOMALY_THRESHOLD,
        "asset_features": {},
        "fallback_mad": {},
    }

    # 5. Calibrate each asset type and feature
    for asset_type, features in ASSET_FEATURES.items():

        asset_data = data[
            data["asset_type"] == asset_type
        ]

        if asset_data.empty:
            continue

        configuration["asset_features"][asset_type] = features
        configuration["fallback_mad"][asset_type] = {}

        for feature in features:

            historical_deviations = []

            for _, asset_group in asset_data.groupby(
                "asset_id", sort=False
            ):

                values = pd.to_numeric(
                    asset_group[feature],
                    errors="coerce"
                )

                # Use only preceding observations
                rolling_median = (
                    values.shift(1)
                    .rolling(
                        window=WINDOW_SIZE,
                        min_periods=MIN_PERIODS
                    )
                    .median()
                )

                deviations = (
                    values - rolling_median
                ).dropna()

                deviations = deviations[
                    np.isfinite(deviations)
                ]

                historical_deviations.extend(
                    deviations.tolist()
                )

            if not historical_deviations:
                raise ValueError(
                    f"Insufficient historical data for "
                    f"{asset_type}.{feature}"
                )

            deviations = np.asarray(
                historical_deviations,
                dtype=float
            )

            # Calculate fallback MAD
            deviation_center = np.median(deviations)

            fallback_mad = np.median(
                np.abs(deviations - deviation_center)
            )

            # Store fallback MAD
            configuration[
                "fallback_mad"
            ][asset_type][feature] = float(fallback_mad)

    # 6. Save detector configuration
    if output_dir is None:

        output_dir = (
            Path(__file__).resolve().parent
            / "model_artifacts"
        )

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    config_path = (
        output_dir / "rolling_mad_config.json"
    )

    with open(
        config_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            configuration,
            file,
            indent=4
        )

    print(
        f"Detector configuration saved to: {config_path}"
    )

    return configuration
