from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd
from pathlib import Path
from typing import Optional

from contextlib import asynccontextmanager

from backend.anomaly_detection.inference import predict_anomalies
from fastapi.encoders import jsonable_encoder

# Dircetory Declaration

ROOT_DIR = Path(__file__).resolve().parents[1]
BACKEND_DIR = Path(__file__).resolve().parent

DATA_DIR = ROOT_DIR / "data"

PROCESSED_DOCUMENTS_DIR = ROOT_DIR / "processed_documents"

DOCLING_DIR = PROCESSED_DOCUMENTS_DIR / "Docling"
CHUNKS_DIR = PROCESSED_DOCUMENTS_DIR / "Chunks"
CHROMA_DB_DIR = PROCESSED_DOCUMENTS_DIR / "ChromaDB"

MODEL_ARTIFACTS_DIR = (
    BACKEND_DIR / "anomaly_detection" / "artifacts"
)


# @asynccontextmanager
# async def lifespan(app: FastAPI):

#     print("Starting Greenko Copilot API...")

#     app.state.history_df = load_history_data()

#     print("Historical telemetry loaded:", app.state.history_df.shape)

#     # Initialize RAG pipeline
#     print("Initializing RAG pipeline...")

#     from backend.rag_system.the_rag_pipeline import RAGPipeline

#     app.state.rag_pipeline = RAGPipeline()

#     print("RAG pipeline initialized successfully.")


#     yield

#     print("Shutting down Greenko Copilot API...")

@asynccontextmanager
async def lifespan(app: FastAPI):

    print("Starting Greenko Copilot API...")

    # Load historical telemetry for ML inference
    app.state.history_df = load_history_data()

    print("Historical telemetry loaded:", app.state.history_df.shape)

    print("ML inference API initialized successfully.")

    yield

    print("Shutting down Greenko Copilot API...")

# FastAPI Intialisation

app = FastAPI(title='Renewable Asset Fault & Anomaly Detection Copilot', description=(
        "An intelligent renewable energy monitoring system "
        "for solar and wind asset anomaly detection, "
        "technical knowledge retrieval, and operator assistance."
    ),
    lifespan = lifespan
)
# BASE = Path(__file__).resolve().parents[1]
# DATA = BASE / 'data'

class TelemetryInput(BaseModel):

    timestamp_id: int

    asset_id: str = Field(
        min_length=1,
        description="Unique renewable energy asset identifier."
    )

    hour: int = Field(
        ge=0,
        le=23,
        description="Hour of the day (0-23)."
    )

    irradiance_wm2: Optional[float] = None

    wind_speed_ms: Optional[float] = None

    temperature_c: Optional[float] = None

    vibration_mm_s: Optional[float] = None

    power_mw: Optional[float] = None


#class Query(BaseModel):
#    text: str = ''


# Basic Endonits


@app.get('/api/demo')
def demo():
    return {
        'case_study': 'Renewable Asset Fault & Anomaly Detection Copilot',
        'status': 'starter app running',
        'expected_outcome': 'A Python ML pipeline that flags anomalies in solar/wind asset telemetry and a copilot UI that explains likely causes and recommended next checks.'
    }

# @app.post("/api/run")
# def run_case(telemetry: TelemetryInput):

#     # 1. Convert incoming telemetry into a DataFrame
#     inference_df = pd.DataFrame(
#         [telemetry.model_dump()]
#     )

#     # 2. Run anomaly detection
#     bundle = predict_anomalies(
#         history_df=app.state.history_df,
#         inference_df=inference_df,
#         config_path=MODEL_ARTIFACTS_DIR / "rolling_mad_config.json",
#         return_bundle=True
#     )

#     # 3. Extract anomaly records
#     anomaly_records = bundle["anomaly_records"]

#     # 4. Check ML inference result

#     if not anomaly_records:

#         # Normal telemetry: RAG is not required
#         normal_records = bundle["normal_df"].to_dict(
#             orient="records"
#         )

#         response = {
#             "status": "normal",
#             "message": "No anomaly detected.",
#             "normal_records": normal_records,
#             "diagnostics": []
#         }

#     else:

#         # Anomaly detected: Invoke RAG
#         diagnostic_results = []

#         for record in anomaly_records:

#             try:
#                 diagnostic = app.state.rag_pipeline.run(record)

#                 diagnostic_results.append({
#                     "asset_id": record.get("asset_id"),
#                     "status": "success",
#                     "diagnostic": diagnostic
#                 })

#             except Exception as exc:

#                 print(f"RAG diagnostic failed: {exc}")

#                 diagnostic_results.append({
#                     "asset_id": record.get("asset_id"),
#                     "status": "failed",
#                     "message": "Diagnostic generation failed."
#                 })

#         response = {
#             "status": "anomaly_detected",
#             "anomaly_count": len(anomaly_records),
#             "anomaly_records": anomaly_records,
#             "diagnostics": diagnostic_results
#         }

#     # 5. Return JSON-compatible response
#     return jsonable_encoder(response)

#     # 4. Prepare response
#     if anomaly_records:

#         response = {
#             "status": "anomaly_detected",
#             "anomaly_count": len(anomaly_records),
#             "anomaly_records": anomaly_records
#         }

#     else:

#         normal_records = bundle["normal_df"].to_dict(
#             orient="records"
#         )

#         response = {
#             "status": "normal",
#             "message": "No anomaly detected.",
#             "normal_records": normal_records
#         }

#     return jsonable_encoder(response)


@app.post("/api/run")
def run_case(telemetry: TelemetryInput):

    # 1. Convert incoming telemetry into a DataFrame
    inference_df = pd.DataFrame(
        [telemetry.model_dump()]
    )

    # 2. Run the existing ML anomaly detection algorithm
    bundle = predict_anomalies(
        history_df=app.state.history_df,
        inference_df=inference_df,
        config_path=MODEL_ARTIFACTS_DIR / "rolling_mad_config.json",
        return_bundle=True
    )

    # 3. Extract ML results
    anomaly_records = bundle["anomaly_records"]

    # 4. Prepare response
    if anomaly_records:

        response = {
            "status": "anomaly_detected",
            "anomaly_count": len(anomaly_records),
            "anomaly_records": anomaly_records
        }

    else:

        normal_records = bundle["normal_df"].to_dict(
            orient="records"
        )

        response = {
            "status": "normal",
            "message": "No anomaly detected.",
            "normal_records": normal_records
        }

    # 5. Return JSON-compatible response
    return jsonable_encoder(response)


def load_history_data():

    dataset_path = DATA_DIR / "asset_telemetry.csv"

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Telemetry dataset not found: {dataset_path}"
        )

    raw_df = pd.read_csv(dataset_path)

    required_columns = [
        "timestamp_id",
        "asset_id",
        "hour",
        "irradiance_wm2",
        "wind_speed_ms",
        "temperature_c",
        "vibration_mm_s",
        "power_mw"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in raw_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing dataset columns: {missing_columns}"
        )

    # Keep only raw telemetry columns
    raw_df = raw_df[required_columns].copy()

    train_parts = []

    # Split chronologically for each asset
    for asset_id, group in raw_df.groupby("asset_id"):

        group = group.sort_values(
            "timestamp_id"
        ).reset_index(drop=True)

        split_index = int(len(group) * 0.8)

        if split_index < 4 or split_index >= len(group):
            raise ValueError(
                f"Insufficient historical observations "
                f"for asset: {asset_id}"
            )

        train_parts.append(
            group.iloc[:split_index]
        )

    history_df = pd.concat(
        train_parts,
        ignore_index=True
    )

    history_df = history_df.sort_values(
        ["asset_id", "timestamp_id"]
    ).reset_index(drop=True)

    return history_df
