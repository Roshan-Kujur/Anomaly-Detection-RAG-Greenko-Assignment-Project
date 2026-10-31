     
"""RAG evaluation utilities for the Greenko anomaly-diagnostics project.

Input records must contain:
    user_input: str                 Diagnostic query sent to the RAG system
    retrieved_contexts: list[str]  Ordered contexts returned by retrieval/reranking
    response: str | dict            Final diagnostic answer from the RAG system

Optional:
    reference: str                  Expert-reviewed reference answer
    record_id, asset_id, asset_type, timestamp_id, anomaly_features, ...

Metrics:
    - Faithfulness: response claims supported by retrieved contexts.
    - Answer Relevancy: response relevance to the diagnostic query.
    - Context Precision: calculated only when an expert reference is supplied.
    - Context Utilization: calculated instead of Context Precision when no
      reference is supplied. These are related but NOT interchangeable metrics.

This module uses the current collections-based Ragas metric API.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


SCORE_COLUMNS = [
    "faithfulness",
    "answer_relevancy",
    "context_precision",
    "context_utilization",
]


def _to_text(value: Any) -> str:
    """Convert string or structured LLM output into text for Ragas."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, ensure_ascii=False, default=str)
    return str(value).strip()


def _context_to_text(context: Any) -> str:
    """Extract textual content from strings, LangChain Documents, or mappings."""
    if isinstance(context, str):
        return context.strip()

    if isinstance(context, Mapping):
        for key in ("page_content", "text", "content", "document"):
            value = context.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(dict(context), ensure_ascii=False, default=str)

    page_content = getattr(context, "page_content", None)
    if isinstance(page_content, str) and page_content.strip():
        return page_content.strip()

    return str(context).strip()


def _normalise_record(record: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Validate and normalize one RAG trace into the Ragas input schema."""
    if not isinstance(record, Mapping):
        raise TypeError(f"Record {index} must be a dictionary-like object.")

    # Accept the canonical name and common trace-field aliases.
    user_input = record.get("user_input")
    if user_input is None:
        user_input = record.get("diagnostic_query", record.get("query"))

    response = record.get("response")
    if response is None:
        response = record.get(
            "generated_response",
            record.get("rag_response", record.get("answer")),
        )

    contexts = record.get("retrieved_contexts")
    if contexts is None:
        contexts = record.get("contexts", record.get("retrieved_documents"))

    user_input = _to_text(user_input)
    response = _to_text(response)

    if not user_input:
        raise ValueError(
            f"Record {index} has no user_input/diagnostic_query."
        )
    if not response:
        raise ValueError(
            f"Record {index} has no response/generated_response."
        )
    if contexts is None:
        raise ValueError(
            f"Record {index} has no retrieved_contexts."
        )

    if isinstance(contexts, (str, Mapping)):
        contexts = [contexts]
    elif not isinstance(contexts, Sequence):
        raise TypeError(
            f"Record {index} retrieved_contexts must be a list of contexts."
        )

    retrieved_contexts = [
        _context_to_text(context)
        for context in contexts
    ]
    retrieved_contexts = [
        context for context in retrieved_contexts if context
    ]

    if not retrieved_contexts:
        raise ValueError(
            f"Record {index} has no non-empty retrieved context."
        )

    reference = _to_text(record.get("reference"))

    record_id = record.get("record_id")
    if record_id is None:
        asset_id = record.get("asset_id", "unknown_asset")
        timestamp_id = record.get("timestamp_id", index)
        record_id = f"{asset_id}_{timestamp_id}"

    normalized = {
        "record_id": str(record_id),
        "user_input": user_input,
        "retrieved_contexts": retrieved_contexts,
        "response": response,
        "reference": reference or None,
    }

    # Retain useful anomaly metadata for slicing and later inspection.
    for key in (
        "asset_id",
        "asset_type",
        "timestamp_id",
        "anomaly_count",
        "anomaly_features",
        "is_anomaly",
    ):
        if key in record:
            normalized[key] = record[key]

    return normalized


def load_evaluation_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Load RAG traces from a JSON file containing a list of records."""
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"RAG evaluation input not found: {input_path}")

    with input_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    # Also support a wrapper object such as {"records": [...]}.
    if isinstance(payload, dict):
        payload = payload.get("records")

    if not isinstance(payload, list) or not payload:
        raise ValueError(
            "Evaluation JSON must contain a non-empty list of RAG records "
            "or an object with a non-empty 'records' list."
        )

    return payload


class GreenkoRAGEvaluator:
    """Run core Ragas metrics over captured Greenko RAG traces."""

    def __init__(
        self,
        evaluator_model: str = "gpt-5.4-mini",
        embedding_model: str = "text-embedding-3-small",
        api_key: str | None = None,
    ) -> None:
        # Load project .env when python-dotenv is available. Existing environment
        # variables take precedence because load_dotenv() does not override them.
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass

        resolved_api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not resolved_api_key:
            raise EnvironmentError(
                "OPENAI_API_KEY is not set. Add it to your .env file or "
                "pass api_key when creating GreenkoRAGEvaluator."
            )

        try:
            from openai import AsyncOpenAI
            from ragas.llms import llm_factory
            from ragas.embeddings.base import embedding_factory
            from ragas.metrics.collections import (
                AnswerRelevancy,
                ContextPrecision,
                ContextUtilization,
                Faithfulness,
            )
        except ImportError as exc:
            raise ImportError(
                "Ragas' collections API and the OpenAI SDK are required. "
                "Install them in the project's active virtual environment with:\n"
                "    pip install -U ragas openai python-dotenv"
            ) from exc

        self.client = AsyncOpenAI(api_key=resolved_api_key)
        self.llm = llm_factory(evaluator_model, client=self.client)
        self.embeddings = embedding_factory(
            "openai",
            model=embedding_model,
            client=self.client,
        )

        self.faithfulness_metric = Faithfulness(llm=self.llm)
        self.answer_relevancy_metric = AnswerRelevancy(
            llm=self.llm,
            embeddings=self.embeddings,
        )
        self.context_precision_metric = ContextPrecision(llm=self.llm)
        self.context_utilization_metric = ContextUtilization(llm=self.llm)

    @staticmethod
    async def _get_score(metric: Any, **kwargs: Any) -> float:
        """Call a Ragas metric and extract its numeric score."""
        result = await metric.ascore(**kwargs)
        value = getattr(result, "value", result)
        if value is None:
            return float("nan")
        return float(value)

    async def evaluate_records(
        self,
        records: Iterable[Mapping[str, Any]],
        output_dir: str | Path | None = None,
        continue_on_metric_error: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Evaluate records sequentially and optionally save detailed results.

        Sequential scoring is intentionally used for a small assignment dataset
        to keep API concurrency and rate-limit pressure low.
        """
        normalized_records = [
            _normalise_record(record, index)
            for index, record in enumerate(records)
        ]
        if not normalized_records:
            raise ValueError("No RAG records were supplied for evaluation.")

        scored_records: list[dict[str, Any]] = []

        for index, record in enumerate(normalized_records, start=1):
            print(
                f"Evaluating record {index}/{len(normalized_records)}: "
                f"{record['record_id']}"
            )

            row = {
                key: value
                for key, value in record.items()
            }
            row.update({column: None for column in SCORE_COLUMNS})
            row["context_metric_used"] = (
                "context_precision"
                if record["reference"]
                else "context_utilization"
            )
            metric_errors: dict[str, str] = {}

            metric_calls = [
                (
                    "faithfulness",
                    self.faithfulness_metric,
                    {
                        "user_input": record["user_input"],
                        "response": record["response"],
                        "retrieved_contexts": record["retrieved_contexts"],
                    },
                ),
                (
                    "answer_relevancy",
                    self.answer_relevancy_metric,
                    {
                        "user_input": record["user_input"],
                        "response": record["response"],
                    },
                ),
            ]

            if record["reference"]:
                metric_calls.append(
                    (
                        "context_precision",
                        self.context_precision_metric,
                        {
                            "user_input": record["user_input"],
                            "reference": record["reference"],
                            "retrieved_contexts": record["retrieved_contexts"],
                        },
                    )
                )
            else:
                metric_calls.append(
                    (
                        "context_utilization",
                        self.context_utilization_metric,
                        {
                            "user_input": record["user_input"],
                            "response": record["response"],
                            "retrieved_contexts": record["retrieved_contexts"],
                        },
                    )
                )

            for metric_name, metric, kwargs in metric_calls:
                try:
                    row[metric_name] = await self._get_score(
                        metric,
                        **kwargs,
                    )
                except Exception as exc:
                    metric_errors[metric_name] = (
                        f"{type(exc).__name__}: {exc}"
                    )
                    if not continue_on_metric_error:
                        raise

            row["metric_errors"] = metric_errors
            row["evaluation_status"] = (
                "success" if not metric_errors else "partial"
            )
            scored_records.append(row)

        results_df = pd.DataFrame(scored_records)
        summary = self._build_summary(results_df)

        if output_dir is not None:
            self._save_results(
                results_df=results_df,
                summary=summary,
                output_dir=output_dir,
            )

        return results_df, summary

    @staticmethod
    def _build_summary(results_df: pd.DataFrame) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "total_records": int(len(results_df)),
            "successful_records": int(
                (results_df["evaluation_status"] == "success").sum()
            ),
            "partially_evaluated_records": int(
                (results_df["evaluation_status"] == "partial").sum()
            ),
            "metrics": {},
        }

        for metric in SCORE_COLUMNS:
            if metric not in results_df.columns:
                continue
            values = pd.to_numeric(
                results_df[metric],
                errors="coerce",
            ).dropna()
            summary["metrics"][metric] = {
                "mean": float(values.mean()) if not values.empty else None,
                "scored_records": int(values.count()),
            }

        summary["context_metric_counts"] = (
            results_df["context_metric_used"]
            .value_counts(dropna=False)
            .to_dict()
        )
        return summary

    @staticmethod
    def _save_results(
        results_df: pd.DataFrame,
        summary: dict[str, Any],
        output_dir: str | Path,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Full JSON preserves contexts, responses, metadata, and metric errors.
        records_path = output_dir / "ragas_scores.json"
        records_df = results_df.copy()
        records_df = records_df.replace({np.nan: None})
        with records_path.open("w", encoding="utf-8") as file:
            json.dump(
                records_df.to_dict(orient="records"),
                file,
                indent=2,
                ensure_ascii=False,
                default=str,
                allow_nan=False,
            )

        # CSV is convenient for spreadsheet inspection.
        csv_path = output_dir / "ragas_scores.csv"
        csv_df = results_df.copy()
        for column in ("retrieved_contexts", "anomaly_features", "metric_errors"):
            if column in csv_df.columns:
                csv_df[column] = csv_df[column].apply(
                    lambda value: json.dumps(value, ensure_ascii=False, default=str)
                    if value is not None else ""
                )
        csv_df.to_csv(csv_path, index=False, encoding="utf-8-sig")

        summary_path = output_dir / "ragas_summary.json"
        with summary_path.open("w", encoding="utf-8") as file:
            json.dump(
                summary,
                file,
                indent=2,
                ensure_ascii=False,
                allow_nan=False,
            )

        print(f"\nDetailed scores saved to: {records_path}")
        print(f"CSV scores saved to: {csv_path}")
        print(f"Aggregate summary saved to: {summary_path}")

    async def evaluate_file(
        self,
        input_path: str | Path,
        output_dir: str | Path,
        max_records: int | None = None,
        continue_on_metric_error: bool = True,
    ) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Load a JSON trace file, evaluate it, and save the results."""
        records = load_evaluation_records(input_path)
        if max_records is not None:
            if max_records <= 0:
                raise ValueError("max_records must be greater than zero.")
            records = records[:max_records]

        return await self.evaluate_records(
            records=records,
            output_dir=output_dir,
            continue_on_metric_error=continue_on_metric_error,
        )

    async def close(self) -> None:
        """Close the underlying async OpenAI client."""
        await self.client.close()


__all__ = [
    "GreenkoRAGEvaluator",
    "load_evaluation_records",
]
