
import json
import math
from typing import Any


class DiagnosticPromptTemplate:
    """
    Builds a concise, evidence-grounded prompt for renewable asset
    anomaly diagnosis.

    Inputs:
        - Diagnostic query
        - Structured diagnostic context
        - Original telemetry
        - Reranked research documents

    Output:
        - system_prompt
        - user_prompt
    """

    def __init__(
        self,
        max_evidence_chunks: int = 5,
        max_chars_per_chunk: int = 2500,
    ):
        self.max_evidence_chunks = max_evidence_chunks
        self.max_chars_per_chunk = max_chars_per_chunk

        self.system_prompt = """
You are a renewable-energy asset diagnostic assistant for Greenko Group.

RULES
1. Diagnose Solar inverters and Wind turbines according to their
   different physical operating characteristics.
2. Use the supplied telemetry, anomaly context, and retrieved research.
   Never invent measurements, evidence, or source citations.
3. Distinguish observed anomalies from possible causes. A retrieved
   document describes general knowledge; it does not prove a fault
   exists in the current asset.
4. Recommend plausible causes only when supported by telemetry,
   engineering reasoning, or retrieved evidence. Explain the supporting
   observations and cite the relevant evidence IDs.
5. If evidence is weak, conflicting, or insufficient, explicitly say so.
   Do not force a single root cause or claim certainty.
6. Provide practical, safe troubleshooting checks and preventive actions.
   Recommend qualified personnel for hazardous or specialized work.
7. Treat missing measurements as unknown, never as zero.
8. Return only valid JSON matching the requested structure.
   Do not include Markdown fences or additional text.
""".strip()

    @staticmethod
    def _make_json_safe(value: Any) -> Any:
        """Convert common non-JSON values into JSON-safe values."""

        if isinstance(value, dict):
            return {
                str(key): DiagnosticPromptTemplate._make_json_safe(item)
                for key, item in value.items()
            }

        if isinstance(value, (list, tuple)):
            return [
                DiagnosticPromptTemplate._make_json_safe(item)
                for item in value
            ]

        if hasattr(value, "item"):
            try:
                value = value.item()
            except (ValueError, TypeError):
                pass

        if isinstance(value, float) and not math.isfinite(value):
            return None

        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        return str(value)

    def _prepare_evidence(self, retrieved_documents: list) -> list:
        """Keep only top-ranked evidence and preserve source metadata."""

        evidence = []

        for index, document in enumerate(
            retrieved_documents[:self.max_evidence_chunks],
            start=1,
        ):
            metadata = document.get("metadata") or {}

            text = (
                document.get("text")
                or document.get("page_content")
                or document.get("content")
                or ""
            ).strip()

            if not text:
                continue

            evidence.append({
                "evidence_id": f"E{index}",
                "source_file": metadata.get(
                    "source_file", "Unknown source"
                ),
                "document_id": metadata.get("document_id"),
                "chunk_id": metadata.get(
                    "chunk_id",
                    document.get("chunk_id"),
                ),
                "asset_type": metadata.get("asset_type"),
                "relevance_score": document.get(
                    "rerank_score",
                    document.get("score"),
                ),
                "text": text[:self.max_chars_per_chunk],
            })

        return self._make_json_safe(evidence)

    def build_prompt(
        self,
        diagnostic_query: str,
        diagnostic_context: dict,
        original_telemetry: dict,
        retrieved_documents: list,
    ) -> dict:
        """Build the final prompts for the diagnostic LLM."""

        if not diagnostic_query or not diagnostic_query.strip():
            raise ValueError("diagnostic_query cannot be empty.")

        if not isinstance(diagnostic_context, dict):
            raise TypeError("diagnostic_context must be a dictionary.")

        if not isinstance(original_telemetry, dict):
            raise TypeError("original_telemetry must be a dictionary.")

        if not retrieved_documents:
            raise ValueError(
                "No retrieved evidence provided. "
                "Check retrieval before generating a diagnosis."
            )

        evidence = self._prepare_evidence(retrieved_documents)

        if not evidence:
            raise ValueError(
                "Retrieved documents contain no usable text."
            )

        payload = {
            "diagnostic_query": diagnostic_query,
            "diagnostic_context": diagnostic_context,
            "original_telemetry": original_telemetry,
            "retrieved_evidence": evidence,
            "required_output_schema": {
                "asset_id": "string or null",
                "asset_type": "Solar or Wind",
                "anomaly_summary": "string",
                "potential_causes": [
                    {
                        "cause": "string",
                        "explanation": "string",
                        "supporting_observations": ["string"],
                        "evidence_sources": ["E1"],
                    }
                ],
                "troubleshooting_checks": [
                    {
                        "check": "string",
                        "reason": "string",
                    }
                ],
                "preventive_recommendations": ["string"],
                "evidence_limitations": ["string"],
                "overall_assessment": "string",
            },
        }

        safe_payload = self._make_json_safe(payload)

        user_prompt = f"""
Analyze this renewable-energy asset anomaly.

Use the supplied evidence IDs (E1, E2, etc.) when citing retrieved
research. Reference only IDs that actually appear in the evidence.
Keep recommendations specific, actionable, and proportional to
the strength of the evidence.

Return the requested JSON object. If the available information
does not support a reliable diagnosis, state the limitations rather
than inventing a cause.

INPUT DATA:
{json.dumps(safe_payload, ensure_ascii=False, separators=(",", ":"))}
""".strip()

        return {
            "system_prompt": self.system_prompt,
            "user_prompt": user_prompt,
        }