
import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.data_pipeline.data_validation import validate_data
from backend.data_pipeline.data_preprocessing import preprocess_data
from backend.data_pipeline.feature_engineering import feature_engineering


# 1. LOAD AND VALIDATE DETECTOR CONFIGURATION


def load_detector_config(config_path=None):

    if config_path is None:
        config_path = (
            Path(__file__).resolve().parent
            / "model_artifacts"
            / "rolling_mad_config.json"
        )

    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {config_path}"
        )

    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)

    # Validate configuration structure
    required_keys = [
        "window_size",
        "min_periods",
        "anomaly_threshold",
        "asset_features",
        "fallback_mad"
    ]

    missing_keys = [
        key for key in required_keys
        if key not in config
    ]

    if missing_keys:
        raise ValueError(
            f"Missing configuration keys: {missing_keys}"
        )

    # Normalize numeric configuration
    config["window_size"] = int(config["window_size"])
    config["min_periods"] = int(config["min_periods"])
    config["anomaly_threshold"] = float(
        config["anomaly_threshold"]
    )

    if config["window_size"] <= 0:
        raise ValueError("window_size must be positive.")

    if not (
        1 <= config["min_periods"]
        <= config["window_size"]
    ):
        raise ValueError("Invalid min_periods configuration.")

    if (
        not np.isfinite(config["anomaly_threshold"])
        or config["anomaly_threshold"] <= 0
    ):
        raise ValueError("Invalid anomaly_threshold.")

    # Normalize asset feature mapping
    normalized_features = {}

    for asset_type, features in config["asset_features"].items():

        asset_type = str(asset_type).strip().lower()

        if asset_type in normalized_features:
            raise ValueError(
                f"Duplicate asset type in configuration: {asset_type}"
            )

        if not isinstance(features, list) or not features:
            raise ValueError(
                f"No valid features configured for {asset_type}."
            )

        normalized_features[asset_type] = [
            str(feature).strip()
            for feature in features
        ]

    # Normalize fallback MAD configuration
    normalized_fallback = {}

    for asset_type, fallback_values in config["fallback_mad"].items():

        asset_type = str(asset_type).strip().lower()

        normalized_fallback[asset_type] = {
            str(feature).strip(): float(value)
            for feature, value in fallback_values.items()
        }

    # Verify that all configured features have fallback MAD values
    for asset_type, features in normalized_features.items():

        if asset_type not in normalized_fallback:
            raise ValueError(
                f"Fallback MAD configuration missing for {asset_type}."
            )

        for feature in features:

            if feature not in normalized_fallback[asset_type]:
                raise ValueError(
                    f"Fallback MAD missing for "
                    f"{asset_type}: {feature}"
                )

            value = normalized_fallback[asset_type][feature]

            if not np.isfinite(value) or value < 0:
                raise ValueError(
                    f"Invalid fallback MAD for {feature}."
                )

    config["asset_features"] = normalized_features
    config["fallback_mad"] = normalized_fallback

    return config


# ==================================================
# 2. CONVERT VALUES INTO JSON-COMPATIBLE TYPES
# ==================================================

def make_json_safe(value):

    if isinstance(value, dict):
        return {
            str(key): make_json_safe(val)
            for key, val in value.items()
        }

    if isinstance(value, (list, tuple, set)):
        return [
            make_json_safe(item)
            for item in value
        ]

    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())

    if value is None or value is pd.NA or value is pd.NaT:
        return None

    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()

    if isinstance(value, np.datetime64):
        if np.isnat(value):
            return None
        return pd.Timestamp(value).isoformat()

    if isinstance(value, (np.bool_, bool)):
        return bool(value)

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, (np.floating, float)):
        return (
            float(value)
            if np.isfinite(value)
            else None
        )

    if isinstance(value, np.generic):
        return make_json_safe(value.item())

    # Handle scalar missing values safely
    try:
        missing = pd.isna(value)

        if isinstance(missing, (bool, np.bool_)):
            if bool(missing):
                return None

    except (TypeError, ValueError):
        pass

    return value


# ==================================================
# 3. PREPARE RAW TELEMETRY
# ==================================================

def normalize_raw_telemetry(df, dataset_name):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{dataset_name} must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            f"{dataset_name} cannot be empty."
        )

    df = df.copy()

    # Normalize column names
    df.columns = df.columns.astype(str).str.strip()

    required_columns = [
        "timestamp_id",
        "asset_id"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )

    # Normalize asset identifiers
    if df["asset_id"].isna().any():
        raise ValueError(
            f"{dataset_name} contains missing asset IDs."
        )

    df["asset_id"] = (
        df["asset_id"]
        .astype(str)
        .str.strip()
    )

    if df["asset_id"].eq("").any():
        raise ValueError(
            f"{dataset_name} contains empty asset IDs."
        )

    # timestamp_id is a sequential identifier
    df["timestamp_id"] = pd.to_numeric(
        df["timestamp_id"],
        errors="coerce"
    )

    if (
        df["timestamp_id"].isna().any()
        or not np.isfinite(df["timestamp_id"]).all()
    ):
        raise ValueError(
            f"{dataset_name} contains invalid timestamp_id values."
        )

    return df


# ==================================================
# 4. COMPLETE DATA PREPARATION PIPELINE
# ==================================================

def prepare_telemetry(df, dataset_name):

    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"{dataset_name} must be a pandas DataFrame."
        )

    if df.empty:
        raise ValueError(
            f"{dataset_name} cannot be empty."
        )

    # Step 1: Data validation
    validation_result = validate_data(df.copy())

    if (
        isinstance(validation_result, (bool, np.bool_))
        and not bool(validation_result)
    ):
        raise ValueError(
            f"{dataset_name} failed data validation."
        )

    # Step 2: Data preprocessing
    processed_df = preprocess_data(df.copy())

    if not isinstance(processed_df, pd.DataFrame):
        raise TypeError(
            "preprocess_data() must return a pandas DataFrame."
        )

    if processed_df.empty:
        raise ValueError(
            f"{dataset_name} is empty after preprocessing."
        )

    # Step 3: Feature engineering
    engineered_df = feature_engineering(processed_df.copy())

    if not isinstance(engineered_df, pd.DataFrame):
        raise TypeError(
            "feature_engineering() must return a pandas DataFrame."
        )

    if engineered_df.empty:
        raise ValueError(
            f"{dataset_name} is empty after feature engineering."
        )

    engineered_df = engineered_df.copy()

    engineered_df.columns = (
        engineered_df.columns.astype(str).str.strip()
    )

    required_output_columns = [
        "timestamp_id",
        "asset_id",
        "asset_type"
    ]

    missing_columns = [
        column
        for column in required_output_columns
        if column not in engineered_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Feature engineering did not produce required columns: "
            f"{missing_columns}"
        )

    return engineered_df.reset_index(drop=True)


# ==================================================
# 5. CREATE OBSERVATION IDENTIFIERS
# ==================================================

def get_record_keys(df):

    asset_ids = (
        df["asset_id"]
        .astype(str)
        .str.strip()
        .tolist()
    )

    timestamps = pd.to_numeric(
        df["timestamp_id"],
        errors="coerce"
    ).tolist()

    return list(zip(asset_ids, timestamps))


# ==================================================
# 6. VALIDATE ENGINEERED DETECTOR INPUTS
# ==================================================

def validate_detector_inputs(df, config):

    required_columns = [
        "timestamp_id",
        "asset_id",
        "asset_type"
    ]

    feature_mapping = config["asset_features"]

    all_features = {
        feature
        for features in feature_mapping.values()
        for feature in features
    }

    required_columns.extend(sorted(all_features))

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Engineered data is missing detector columns: "
            f"{missing_columns}"
        )

    if df[
        ["timestamp_id", "asset_id", "asset_type"]
    ].isna().any().any():

        raise ValueError(
            "Engineered data contains missing identifiers."
        )

    # Normalize asset types
    asset_types = (
        df["asset_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    supported_types = set(feature_mapping.keys())

    unknown_types = set(asset_types.unique()) - supported_types

    if unknown_types:
        raise ValueError(
            f"Unsupported asset types: {sorted(unknown_types)}. "
            f"Configured types: {sorted(supported_types)}"
        )

    return feature_mapping


# ==================================================
# 7. CALCULATE ROLLING MEDIAN / MAD ANOMALY SCORES
# ==================================================

def calculate_anomaly_scores(
    asset_df,
    asset_type,
    config
):

    window = config["window_size"]
    min_periods = config["min_periods"]
    threshold = config["anomaly_threshold"]

    asset_type = (
        str(asset_type)
        .strip()
        .lower()
    )

    features = config["asset_features"][asset_type]

    fallback_values = config["fallback_mad"][asset_type]

    result = asset_df.copy()

    # Ensure chronological order
    result = result.sort_values(
        "timestamp_id"
    ).copy()

    for feature in features:

        values = pd.to_numeric(
            result[feature],
            errors="coerce"
        )

        # Use preceding observations only
        previous_values = values.shift(1)

        # Calculate rolling median
        rolling_median = previous_values.rolling(
            window=window,
            min_periods=min_periods
        ).median()

        # Calculate rolling MAD
        rolling_mad = previous_values.rolling(
            window=window,
            min_periods=min_periods
        ).apply(
            lambda x: np.median(
                np.abs(x - np.median(x))
            ),
            raw=True
        )

        # Deviation from historical median
        deviation = values - rolling_median

        fallback_mad = fallback_values[feature]

        # Substitute fallback only where rolling MAD is zero
        effective_mad = rolling_mad.copy()

        zero_mad = (
            rolling_mad.notna()
            & np.isclose(rolling_mad, 0.0)
        )

        if fallback_mad > 0:

            effective_mad.loc[zero_mad] = fallback_mad

        # Modified Z-score
        modified_zscore = (
            0.6745 * deviation
            / effective_mad.where(effective_mad > 0)
        )

        # Special handling when both rolling and fallback MAD are zero
        zero_mad_anomaly = (
            zero_mad
            & (fallback_mad == 0)
            & deviation.notna()
            & (deviation.abs() > 1e-12)
        )

        # Feature-level anomaly classification
        feature_anomaly = (
            modified_zscore.abs() > threshold
        ) | zero_mad_anomaly

        # Store all feature-level outputs
        result[f"{feature}_rolling_median"] = rolling_median

        result[f"{feature}_rolling_mad"] = rolling_mad

        result[f"{feature}_median_deviation"] = deviation

        result[f"{feature}_modified_zscore"] = modified_zscore

        result[f"{feature}_is_anomaly"] = (
            feature_anomaly
            .fillna(False)
            .astype(bool)
        )

    return result


# ==================================================
# 8. BUILD STRUCTURED RAG-READY ANOMALY RECORDS
# ==================================================

def build_anomaly_record(row):

    telemetry = row.to_dict()

    anomaly_features = telemetry.get(
        "anomaly_features",
        []
    )

    feature_deviations = {}

    for feature in anomaly_features:

        feature_deviations[feature] = {

            "observed_value": telemetry.get(feature),

            "rolling_median": telemetry.get(
                f"{feature}_rolling_median"
            ),

            "rolling_mad": telemetry.get(
                f"{feature}_rolling_mad"
            ),

            "median_deviation": telemetry.get(
                f"{feature}_median_deviation"
            ),

            "modified_zscore": telemetry.get(
                f"{feature}_modified_zscore"
            )
        }

    # Keep telemetry and anomaly fields at the top level.
    # This maintains compatibility with RAGPipeline.run(telemetry).

    telemetry["anomaly_details"] = {

        "is_anomaly": True,

        "anomaly_count": int(
            telemetry["anomaly_count"]
        ),

        "anomaly_features": anomaly_features,

        "feature_deviations": feature_deviations
    }

    return make_json_safe(telemetry)


# ==================================================
# 9. MAIN INFERENCE PIPELINE
# ==================================================

def predict_anomalies(
    history_df,
    inference_df,
    config_path=None,
    return_bundle=True
):

    """
    Complete anomaly detection pipeline.

    Inputs:
        history_df:
            Raw historical telemetry.

        inference_df:
            Raw incoming telemetry.

    Internal workflow:
        Data validation
        -> Data preprocessing
        -> Feature engineering
        -> Rolling Median/MAD scoring
        -> Feature-level anomaly detection
        -> Asset-level classification
        -> RAG-ready anomaly records.

    Outputs:
        predictions_df
        anomalies_df
        normal_df
        anomaly_records
    """

    # --------------------------------------------------
    # 1. Load trained detector configuration
    # --------------------------------------------------

    config = load_detector_config(config_path)

    # --------------------------------------------------
    # 2. Validate and normalize raw inputs
    # --------------------------------------------------

    history = normalize_raw_telemetry(
        history_df,
        "Historical data"
    )

    incoming = normalize_raw_telemetry(
        inference_df,
        "Inference data"
    )

    # --------------------------------------------------
    # 3. Verify observation uniqueness
    # --------------------------------------------------

    history_keys = get_record_keys(history)
    incoming_keys = get_record_keys(incoming)

    if len(set(history_keys)) != len(history_keys):
        raise ValueError(
            "Duplicate asset_id and timestamp_id combinations "
            "found in historical data."
        )

    if len(set(incoming_keys)) != len(incoming_keys):
        raise ValueError(
            "Duplicate asset_id and timestamp_id combinations "
            "found in incoming data."
        )

    all_keys = history_keys + incoming_keys

    if len(set(all_keys)) != len(all_keys):
        raise ValueError(
            "An asset and timestamp combination occurs in both "
            "historical and incoming data."
        )

    history_asset_ids = set(
        history["asset_id"].unique()
    )

    incoming_asset_ids = set(
        incoming["asset_id"].unique()
    )

    assets_without_history = (
        incoming_asset_ids - history_asset_ids
    )

    if assets_without_history:
        raise ValueError(
            "Historical observations are missing for assets: "
            f"{sorted(assets_without_history)}"
        )

    # Check chronological ordering for every incoming asset
    history_max = (
        history.groupby("asset_id")["timestamp_id"]
        .max()
    )

    incoming_min = (
        incoming.groupby("asset_id")["timestamp_id"]
        .min()
    )

    for asset_id in incoming_asset_ids:

        if history_max[asset_id] >= incoming_min[asset_id]:

            raise ValueError(
                f"Historical observations must precede incoming "
                f"observations for {asset_id}."
            )

    # --------------------------------------------------
    # 4. Combine history and incoming data
    # --------------------------------------------------

    combined_raw = pd.concat(
        [history, incoming],
        ignore_index=True
    )

    combined_raw = combined_raw.sort_values(
        ["asset_id", "timestamp_id"]
    ).reset_index(drop=True)

    incoming_key_set = set(incoming_keys)

    expected_key_set = set(all_keys)

    # --------------------------------------------------
    # 5. Run data validation, preprocessing and
    #    feature engineering ONCE on combined data
    # --------------------------------------------------

    prepared_df = prepare_telemetry(
        combined_raw,
        "Combined historical and incoming telemetry"
    )

    # Normalize identifiers after data preparation
    prepared_df["asset_id"] = (
        prepared_df["asset_id"]
        .astype(str)
        .str.strip()
    )

    prepared_df["timestamp_id"] = pd.to_numeric(
        prepared_df["timestamp_id"],
        errors="coerce"
    )

    if (
        prepared_df["timestamp_id"].isna().any()
        or not np.isfinite(prepared_df["timestamp_id"]).all()
    ):
        raise ValueError(
            "Feature engineering produced invalid timestamp_id values."
        )

    # --------------------------------------------------
    # 6. Verify that preparation preserved observations
    # --------------------------------------------------

    prepared_keys = get_record_keys(prepared_df)

    if len(set(prepared_keys)) != len(prepared_keys):
        raise ValueError(
            "Duplicate asset/timestamp combinations appeared "
            "after data preparation."
        )

    prepared_key_set = set(prepared_keys)

    missing_keys = expected_key_set - prepared_key_set

    if missing_keys:
        raise ValueError(
            "Data preparation did not preserve all input observations. "
            f"Missing records: {list(missing_keys)[:10]}"
        )

    # --------------------------------------------------
    # 7. Identify incoming observations
    # --------------------------------------------------

    prepared_df["_is_inference"] = [
        key in incoming_key_set
        for key in prepared_keys
    ]

    prepared_df["asset_type"] = (
        prepared_df["asset_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------
    # 8. Validate engineered detector inputs
    # --------------------------------------------------

    feature_mapping = validate_detector_inputs(
        prepared_df,
        config
    )

    # --------------------------------------------------
    # 9. Sort observations chronologically
    # --------------------------------------------------

    prepared_df = prepared_df.sort_values(
        ["asset_id", "timestamp_id"]
    ).reset_index(drop=True)

    results = []

    # --------------------------------------------------
    # 10. Process each asset independently
    # --------------------------------------------------

    for asset_id, asset_df in prepared_df.groupby(
        "asset_id",
        sort=False
    ):

        asset_df = asset_df.copy()

        if asset_df["asset_type"].nunique() != 1:
            raise ValueError(
                f"Inconsistent asset type for {asset_id}."
            )

        asset_type = asset_df["asset_type"].iloc[0]

        historical = asset_df[
            ~asset_df["_is_inference"]
        ]

        incoming_asset = asset_df[
            asset_df["_is_inference"]
        ]

        # Skip assets that have no incoming observations
        if incoming_asset.empty:
            continue

        if historical.empty:
            raise ValueError(
                f"No historical data available for {asset_id}."
            )

        if (
            historical["timestamp_id"].max()
            >= incoming_asset["timestamp_id"].min()
        ):
            raise ValueError(
                f"Historical data must precede inference data "
                f"for {asset_id}."
            )

        # Calculate scores using historical and incoming rows together
        scored_asset = calculate_anomaly_scores(
            asset_df,
            asset_type,
            config
        )

        # Retain only incoming observations
        predictions = scored_asset[
            scored_asset["_is_inference"]
        ].copy()

        results.append(predictions)

    if not results:
        raise ValueError(
            "No inference observations were processed."
        )

    # --------------------------------------------------
    # 11. Combine predictions
    # --------------------------------------------------

    predictions_df = pd.concat(
        results,
        ignore_index=True
    )

    predictions_df = predictions_df.reset_index(
        drop=True
    ).copy()

    # --------------------------------------------------
    # 12. Generate feature-level and asset-level summaries
    # --------------------------------------------------

    asset_types = (
        predictions_df["asset_type"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # Initialize output columns
    predictions_df["anomaly_count"] = 0

    predictions_df["anomaly_features"] = [
        [] for _ in range(len(predictions_df))
    ]

    # IMPORTANT:
    # Process only features belonging to each asset type.
    # Do not create one global list of solar and wind flags.

    for asset_type, features in feature_mapping.items():

        # Select predictions for this asset type
        mask = asset_types == asset_type

        if not mask.any():
            continue

        # Generate expected anomaly columns for this asset type only
        anomaly_columns = [
            f"{feature}_is_anomaly"
            for feature in features
        ]

        # Verify that scoring created all relevant flags
        missing_columns = [
            column
            for column in anomaly_columns
            if column not in predictions_df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Anomaly scoring failed for {asset_type}. "
                f"Missing flags: {missing_columns}. "
                "Check calculate_anomaly_scores() and "
                "the configured asset features."
            )

        # Extract flags for the current asset type
        flags_df = (
            predictions_df.loc[mask, anomaly_columns]
            .fillna(False)
            .astype(bool)
        )

        # Count anomalous features
        predictions_df.loc[mask, "anomaly_count"] = (
            flags_df.sum(axis=1).astype(int)
        )

        # Identify anomalous feature names
        for index, row in flags_df.iterrows():

            predictions_df.at[index, "anomaly_features"] = [
                feature
                for feature in features
                if row[f"{feature}_is_anomaly"]
            ]

    # Overall anomaly classification
    predictions_df["is_anomaly"] = (
        predictions_df["anomaly_count"] > 0
    )

    # --------------------------------------------------
    # 13. Clean and sort predictions
    # --------------------------------------------------

    predictions_df = predictions_df.drop(
        columns=["_is_inference"],
        errors="ignore"
    )

    predictions_df = predictions_df.sort_values(
        ["asset_id", "timestamp_id"]
    ).reset_index(drop=True)

    # --------------------------------------------------
    # 14. Separate anomalous and normal observations
    # --------------------------------------------------

    anomalies_df = predictions_df[
        predictions_df["is_anomaly"]
    ].copy()

    normal_df = predictions_df[
        ~predictions_df["is_anomaly"]
    ].copy()

    # --------------------------------------------------
    # 15. Build RAG-ready anomaly records
    # --------------------------------------------------

    anomaly_records = [
        build_anomaly_record(row)
        for _, row in anomalies_df.iterrows()
    ]

    # --------------------------------------------------
    # 16. Return inference results
    # --------------------------------------------------

    if return_bundle:

        return {
            "predictions_df": predictions_df,
            "anomalies_df": anomalies_df,
            "normal_df": normal_df,
            "anomaly_records": anomaly_records
        }

    return predictions_df