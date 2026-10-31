# Model validation and evaluation pipeline

import json
from pathlib import Path

import numpy as np
import pandas as pd

from backend.anomaly_detection.inference import predict_anomalies


# --------------------------------------------------
# 1. Calculate evaluation metrics
# --------------------------------------------------

def calculate_metrics(y_true, y_pred):
    """Calculate binary classification metrics."""

    y_true = np.asarray(y_true, dtype=bool)
    y_pred = np.asarray(y_pred, dtype=bool)

    tp = int(np.sum(y_true & y_pred))
    tn = int(np.sum(~y_true & ~y_pred))
    fp = int(np.sum(~y_true & y_pred))
    fn = int(np.sum(y_true & ~y_pred))

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    accuracy = (
        (tp + tn) / len(y_true)
        if len(y_true)
        else 0.0
    )

    return {
        "total_observations": len(y_true),
        "true_positives": tp,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
    }


# --------------------------------------------------
# 2. Chronological holdout evaluation
# --------------------------------------------------

def evaluate_holdout(predictions_df):
    """
    Summarize anomaly predictions on the chronological
    holdout dataset.

    These are descriptive statistics, not supervised
    accuracy or precision/recall measurements.
    """

    if predictions_df.empty:
        raise ValueError("Predictions dataframe is empty.")

    total = len(predictions_df)
    anomaly_count = int(predictions_df["is_anomaly"].sum())

    summary = {
        "total_observations": total,
        "detected_anomalies": anomaly_count,
        "normal_observations": total - anomaly_count,
        "anomaly_percentage": round(
            100 * anomaly_count / total, 2
        ),
        "anomalies_by_asset": {},
        "anomalies_by_sensor": {},
    }

    # Anomaly counts for individual assets
    asset_summary = (
        predictions_df.groupby("asset_id")["is_anomaly"]
        .agg(["count", "sum"])
    )

    for asset_id, row in asset_summary.iterrows():
        summary["anomalies_by_asset"][str(asset_id)] = {
            "observations": int(row["count"]),
            "anomalies": int(row["sum"]),
            "anomaly_percentage": round(
                100 * row["sum"] / row["count"], 2
            ),
        }

    # Count sensor-level anomaly flags
    sensor_columns = [
        col for col in predictions_df.columns
        if col.endswith("_is_anomaly")
    ]

    for column in sensor_columns:
        count = int(predictions_df[column].sum())

        summary["anomalies_by_sensor"][
            column.replace("_is_anomaly", "")
        ] = count

    return summary


# --------------------------------------------------
# 3. Inject controlled synthetic anomalies
# --------------------------------------------------

def inject_synthetic_anomalies(test_df, config_path=None):
    """
    Create a separate synthetic dataset with controlled
    power drops and vibration spikes.

    Original input data remains unchanged.
    """

    if test_df.empty:
        raise ValueError("Test dataframe is empty.")

    synthetic_df = test_df.copy()
    synthetic_df["_synthetic_anomaly"] = False

    # Inject one controlled anomaly per eligible asset.
    # Select the middle observation to reduce the chance
    # of choosing a test-set boundary observation.

    for asset_id, group in synthetic_df.groupby("asset_id"):

        indices = group.index.tolist()

        if not indices:
            continue

        target_index = indices[len(indices) // 2]

        asset_type = str(
            synthetic_df.loc[target_index, "asset_type"]
        ).strip().lower()

        # Inject a substantial power drop
        if "power_mw" in synthetic_df.columns:

            current_power = pd.to_numeric(
                synthetic_df.loc[target_index, "power_mw"],
                errors="coerce"
            )

            if pd.notna(current_power):

                if current_power > 0:
                    synthetic_df.loc[
                        target_index, "power_mw"
                    ] = current_power * 0.05
                else:
                    synthetic_df.loc[
                        target_index, "power_mw"
                    ] = current_power - max(
                        abs(current_power), 1.0
                    )

        # Inject a vibration spike
        if "vibration_mm_s" in synthetic_df.columns:

            vibration = pd.to_numeric(
                group["vibration_mm_s"],
                errors="coerce"
            ).dropna()

            current_vibration = pd.to_numeric(
                synthetic_df.loc[
                    target_index, "vibration_mm_s"
                ],
                errors="coerce"
            )

            if pd.notna(current_vibration):

                if len(vibration) > 1:
                    spread = (
                        vibration.quantile(0.75)
                        - vibration.quantile(0.25)
                    )
                else:
                    spread = 0.0

                # Ensure a meaningful artificial increase
                spike_size = max(
                    abs(float(spread)) * 6,
                    abs(float(current_vibration)) * 2,
                    1.0,
                )

                synthetic_df.loc[
                    target_index, "vibration_mm_s"
                ] = current_vibration + spike_size

        synthetic_df.loc[
            target_index, "_synthetic_anomaly"
        ] = True

    return synthetic_df


# --------------------------------------------------
# 4. Evaluate synthetic anomaly detection
# --------------------------------------------------

def evaluate_synthetic_anomalies(
    train_df,
    test_df,
    config_path=None,
):
    """
    Evaluate the detector against artificially injected
    anomalies using the existing inference pipeline.
    """

    synthetic_df = inject_synthetic_anomalies(
        test_df,
        config_path=config_path
    )

    # Keep labels separate from the inference inputs
    y_true = synthetic_df["_synthetic_anomaly"].to_numpy()

    inference_input = synthetic_df.drop(
        columns=["_synthetic_anomaly"]
    )

    # Use original training observations as historical context
    predictions = predict_anomalies(
        history_df=train_df,
        inference_df=inference_input,
        config_path=config_path,
    )

    # Predictions and test rows are ordered by asset/timestamp
    # by inference. Align labels using stable observation keys.
    label_df = synthetic_df[
        ["asset_id", "timestamp_id", "_synthetic_anomaly"]
    ]

    aligned = predictions.merge(
        label_df,
        on=["asset_id", "timestamp_id"],
        how="left",
        validate="one_to_one",
    )

    if aligned["_synthetic_anomaly"].isna().any():
        raise ValueError(
            "Could not align synthetic labels with predictions."
        )

    y_true = aligned["_synthetic_anomaly"].to_numpy(dtype=bool)
    y_pred = aligned["is_anomaly"].to_numpy(dtype=bool)

    metrics = calculate_metrics(y_true, y_pred)

    metrics["synthetic_anomaly_observations"] = int(y_true.sum())
    metrics["synthetic_anomaly_detection_rate"] = round(
        float(y_pred[y_true].mean()), 4
    ) if y_true.any() else None

    metrics["interpretation"] = (
        "Metrics are measured against artificially injected "
        "anomalies. Unmodified observations are treated as "
        "negative examples for this synthetic benchmark; "
        "they are not confirmed normal observations."
    )

    return metrics, aligned


# --------------------------------------------------
# 5. Run and save model evaluation
# --------------------------------------------------

def run_model_validation(
    train_df,
    test_df,
    predictions_df,
    output_dir=None,
    config_path=None,
):
    """
    Run holdout summaries and synthetic anomaly evaluation.
    Save evaluation results for assignment documentation.
    """

    holdout_results = evaluate_holdout(predictions_df)

    synthetic_metrics, synthetic_predictions = (
        evaluate_synthetic_anomalies(
            train_df=train_df,
            test_df=test_df,
            config_path=config_path,
        )
    )

    results = {
        "method": "Rolling Median/MAD",
        "holdout_evaluation": holdout_results,
        "synthetic_anomaly_evaluation": synthetic_metrics,
        "limitations": [
            "The original dataset has no confirmed anomaly labels.",
            "Holdout anomaly counts are descriptive, not supervised metrics.",
            "Synthetic metrics measure detection of injected patterns only.",
            "Unmodified observations are not guaranteed to be normal.",
        ],
    }

    if output_dir is None:
        output_dir = (
            Path(__file__).resolve().parent
            / "model_artifacts"
            / "evaluation"
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary metrics
    with open(
        output_dir / "evaluation_results.json",
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(results, file, indent=4)

    # Save row-level synthetic predictions and labels
    synthetic_predictions.to_csv(
        output_dir / "synthetic_predictions.csv",
        index=False,
    )

    print(f"Evaluation results saved to: {output_dir}")

    return results, synthetic_predictions