

from typing import Any, Mapping


class DiagnosticQueryTransformer:

    def __init__(self):

        self.common_features = [
            "temperature_c",
            "vibration_mm_s",
            "power_mw",
            "power_mw_rolling_median",
            "power_mw_rolling_mad"
        ]

        self.asset_features = {
            "Wind": [
                "wind_speed_ms",
                "wind_power_per_speed"
            ],
            "Solar": [
                "irradiance_wm2",
                "solar_power_per_irradiance"
            ]
        }

        self.feature_labels = {
            "temperature_c": "Temperature",
            "vibration_mm_s": "Vibration",
            "power_mw": "Power output",
            "power_mw_rolling_median": "Rolling median power",
            "power_mw_rolling_mad": "Rolling median absolute deviation",
            "wind_speed_ms": "Wind speed",
            "wind_power_per_speed": "Wind power per unit wind speed",
            "irradiance_wm2": "Solar irradiance",
            "solar_power_per_irradiance": "Solar power per unit irradiance"
        }

    def _get_asset_type(self, telemetry: Mapping[str, Any]) -> str:

        asset_type = telemetry.get("asset_type")

        # Normalize asset_type if it is available.
        if asset_type is not None:
            asset_type = str(asset_type).strip().capitalize()

            if asset_type in self.asset_features:
                return asset_type

        # Derive asset_type from asset_id if missing or invalid.
        asset_id = str(telemetry.get("asset_id", "")).strip().upper()

        if asset_id.startswith("WTG"):
            return "Wind"

        if asset_id.startswith("SOLAR"):
            return "Solar"

        raise ValueError(
            f"Unable to identify asset type. "
            f"Received asset_type={telemetry.get('asset_type')}, "
            f"asset_id={telemetry.get('asset_id')}. "
            f"Expected Wind/Solar or a valid WTG/SOLAR asset ID."
        )

    def transform(
        self,
        telemetry: Mapping[str, Any],
        anomaly_details: Mapping[str, Any] | None = None
    ) -> dict:

        if not telemetry:
            raise ValueError("Telemetry cannot be empty.")

        asset_type = self._get_asset_type(telemetry)

        asset_id = telemetry.get("asset_id", "Unknown asset")

        relevant_features = (
            self.common_features
            + self.asset_features[asset_type]
        )

        selected_features = {}

        for feature in relevant_features:

            value = telemetry.get(feature)

            # Ignore missing values.
            if value is None or str(value).strip().lower() in {
                "", "nan", "none"
            }:
                continue

            selected_features[feature] = value

        if not selected_features:
            raise ValueError(
                "No relevant telemetry features found."
            )

        observations = []

        for feature, value in selected_features.items():

            label = self.feature_labels[feature]

            observations.append(f"{label}: {value}")

        if anomaly_details:

            for feature, value in anomaly_details.items():

                if value is not None and str(value).strip().lower() not in {
                    "", "nan", "none"
                }:
                    observations.append(
                        f"{feature.replace('_', ' ').title()}: {value}"
                    )

        observation_text = "; ".join(observations)

        diagnostic_query = (
            f"Investigate potential operational anomalies in a "
            f"{asset_type} renewable energy asset. "
            f"Observed operating conditions: {observation_text}. "
            f"Identify relevant potential failure mechanisms, "
            f"their relationship with the observed conditions, "
            f"and recommended troubleshooting and preventive "
            f"maintenance checks."
        )

        diagnostic_context = {
            "asset_id": asset_id,
            "asset_type": asset_type,
            "timestamp_id": telemetry.get("timestamp_id"),
            "hour": telemetry.get("hour"),
            "selected_features": selected_features,
            "anomaly_details": dict(anomaly_details or {})
        }

        return {
            "diagnostic_query": diagnostic_query,
            "diagnostic_context": diagnostic_context
        }
