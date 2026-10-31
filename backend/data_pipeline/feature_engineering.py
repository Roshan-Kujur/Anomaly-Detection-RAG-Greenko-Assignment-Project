
import pandas as pd
import numpy as np
from pathlib import Path


# --------------------------------------------------
# 1. Validate required input columns
# --------------------------------------------------

def validate_feature_inputs(df):

    required_columns = [
    "timestamp_id",
    "asset_id",
    "hour",
    "power_mw",
    "vibration_mm_s",
    "irradiance_wm2",
    "wind_speed_ms"
]

    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    if not pd.api.types.is_numeric_dtype(df["timestamp_id"]):
        raise TypeError(
            "timestamp_id must remain numeric."
        )

    if df["asset_id"].isna().any():
        raise ValueError(
            "asset_id contains missing values."
        )


# --------------------------------------------------
# 2. Create asset type
# --------------------------------------------------

def add_asset_type(df):

    asset_id = df["asset_id"].astype(str).str.upper()

    df["asset_type"] = np.select(
        [
            asset_id.str.startswith("SOLAR"),
            asset_id.str.startswith("WTG")
        ],
        [
            "Solar",
            "Wind"
        ],
        default="Unknown"
    )

    if df["asset_type"].eq("Unknown").any():
        unknown_assets = (
            df.loc[
                df["asset_type"] == "Unknown",
                "asset_id"
            ]
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Unknown asset IDs: {unknown_assets}"
        )

    return df


# --------------------------------------------------
# 3. Generate cyclical time features
# --------------------------------------------------

def add_time_features(df):

    df["hour_sin"] = np.sin(
        2 * np.pi * df["hour"] / 24
    )

    df["hour_cos"] = np.cos(
        2 * np.pi * df["hour"] / 24
    )

    df.drop(
        columns=["hour"],
        inplace=True
    )

    return df


# --------------------------------------------------
# 4. Calculate historical rolling statistics
# --------------------------------------------------

def add_rolling_statistics(
    df,
    window=6,
    min_periods=3
):

    columns_to_monitor = [
        "power_mw",
        "vibration_mm_s",
        "irradiance_wm2",
        "wind_speed_ms"
    ]

    # if "temperature_c" in df.columns:
    #     columns_to_monitor.append("temperature_c")

    for col in columns_to_monitor:

        # Historical rolling median
        median_col = f"{col}_rolling_median"

        df[median_col] = (
            df.groupby("asset_id")[col]
            .transform(
                lambda s: s.shift(1)
                .rolling(
                    window=window,
                    min_periods=min_periods
                )
                .median()
            )
        )

        # Historical rolling MAD
        mad_col = f"{col}_rolling_mad"

        df[mad_col] = (
            df.groupby("asset_id")[col]
            .transform(
                lambda s: s.shift(1)
                .rolling(
                    window=window,
                    min_periods=min_periods
                )
                .apply(
                    lambda x: np.median(
                        np.abs(x - np.median(x))
                    ),
                    raw=True
                )
            )
        )

        # Deviation from historical median
        deviation_col = f"{col}_median_deviation"

        df[deviation_col] = (
            df[col] - df[median_col]
        )

        # Modified Z-score
        zscore_col = f"{col}_modified_zscore"

        df[zscore_col] = (
            0.6745
            * df[deviation_col]
            / df[mad_col].replace(0, np.nan)
        )

        # Handle zero MAD
        zero_mad = df[mad_col].eq(0)

        same_as_median = (
            zero_mad
            & df[col].eq(df[median_col])
        )

        df.loc[same_as_median, zscore_col] = 0

        # Flag zero MAD with a different current reading
        zero_mad_flag = f"{col}_zero_mad_deviation"

        df[zero_mad_flag] = (
            zero_mad
            & df[deviation_col].ne(0)
            & df[median_col].notna()
        ).astype(int)

    return df


# --------------------------------------------------
# 5. Calculate power change
# --------------------------------------------------

def add_power_change(df):

    df["power_change_mw"] = (
        df.groupby("asset_id")["power_mw"]
        .diff()
    )

    return df


# --------------------------------------------------
# 6. Complete feature engineering pipeline
# --------------------------------------------------

def feature_engineering(
    df,
    window=6,
    min_periods=3
):

    # Validate input
    validate_feature_inputs(df)

    # Independent copy
    engg_df = df.copy()

    # Sort chronologically
    engg_df = engg_df.sort_values(
        by=["asset_id", "timestamp_id"]
    ).reset_index(drop=True)

    # Create asset type
    engg_df = add_asset_type(engg_df)

    # Generate cyclical time features
    engg_df = add_time_features(engg_df)

    # Generate rolling statistics and anomaly scores
    engg_df = add_rolling_statistics(
        engg_df,
        window=window,
        min_periods=min_periods
    )

    # Calculate power changes
    engg_df = add_power_change(engg_df)

    return engg_df


# --------------------------------------------------
# 7. Export engineered dataset
# --------------------------------------------------

def save_engineered_data(
    engg_df,
    output_path="data/engineered_data.csv"
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    engg_df.to_csv(
        output_path,
        index=False
    )

    print(
        "Engineered data saved successfully:",
        output_path
    )


# --------------------------------------------------
# 8. Execute feature engineering
# --------------------------------------------------

if __name__ == "__main__":

    df = pd.read_csv(
        "data/preprocessed_data.csv"
    )

    engg_df = feature_engineering(
        df,
        window=6,
        min_periods=3
    )

    print("\nAsset type distribution:")
    print(engg_df["asset_type"].value_counts())

    print("\nEngineered dataset shape:")
    print(engg_df.shape)

    save_engineered_data(engg_df)
