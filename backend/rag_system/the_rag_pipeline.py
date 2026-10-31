

class RAGPipeline:
    """
    Orchestrates the Greenko diagnostic RAG workflow.

    Input:
        One anomalous inference-result record.

    Output:
        Structured diagnostic assessment for FastAPI and Copilot UI.
    """

    def __init__(self):
        from backend.rag_system.query_transformer import (
            DiagnosticQueryTransformer
        )
        from backend.rag_system.hybrid_retriever import HybridRetriever
        from backend.rag_system.reranker import CrossEncoderReranker
        from backend.rag_system.prompt_template import DiagnosticPromptTemplate
        from backend.rag_system.answer_generator import DiagnosticGenerator

        self.query_transformer = DiagnosticQueryTransformer()
        self.retriever = HybridRetriever(
            rrf_k=60,
            candidate_k=20
        )
        self.reranker = CrossEncoderReranker()
        self.prompt_builder = DiagnosticPromptTemplate(
            max_evidence_chunks=5,
            max_chars_per_chunk=2500
        )
        self.generator = DiagnosticGenerator()


    # 1. Constructing anomaly details from inference output
    

    def _build_anomaly_details(self, telemetry):

        anomaly_features = telemetry.get("anomaly_features", [])

        anomaly_details = {
            "is_anomaly": bool(telemetry.get("is_anomaly", False)),
            "anomaly_count": int(telemetry.get("anomaly_count", 0)),
            "anomaly_features": anomaly_features,
            "feature_deviations": {}
        }

        for feature in anomaly_features:

            anomaly_details["feature_deviations"][feature] = {
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

        return anomaly_details

    # --------------------------------------------------
    # 2. Execute complete RAG workflow
    # --------------------------------------------------

    def run(self, telemetry: dict, return_trace: bool = False) -> dict:

        if not isinstance(telemetry, dict) or not telemetry:
            raise ValueError("Valid inference telemetry is required.")

        anomaly_details = self._build_anomaly_details(telemetry)

        if not anomaly_details["is_anomaly"]:
            raise ValueError(
                "RAG pipeline accepts only anomalous telemetry."
            )

        if not anomaly_details["anomaly_features"]:
            raise ValueError(
                "Anomaly detected, but no anomalous features were provided."
            )

        # --------------------------------------------------
        # 1. Transform telemetry into a diagnostic query
        # --------------------------------------------------

        transformed = self.query_transformer.transform(
            telemetry=telemetry,
            anomaly_details=anomaly_details
        )

        diagnostic_query = transformed["diagnostic_query"]
        diagnostic_context = transformed["diagnostic_context"]

        asset_type = diagnostic_context["asset_type"]

        # --------------------------------------------------
        # 2. Retrieve relevant research documents
        # --------------------------------------------------

        retrieved_documents = self.retriever.retrieve(
            query=diagnostic_query,
            top_k=20,
            asset_type=asset_type
        )

        if not retrieved_documents:
            raise RuntimeError(
                "No relevant research evidence was retrieved."
            )

        # --------------------------------------------------
        # 3. Rerank retrieved documents
        # --------------------------------------------------

        reranked_documents = self.reranker.rerank(
            query=diagnostic_query,
            documents=retrieved_documents,
            top_k=5
        )

        if not reranked_documents:
            raise RuntimeError(
                "No documents remained after reranking."
            )

        # --------------------------------------------------
        # 4. Build final LLM prompts
        # --------------------------------------------------

        prompts = self.prompt_builder.build_prompt(
            diagnostic_query=diagnostic_query,
            diagnostic_context=diagnostic_context,
            original_telemetry=telemetry,
            retrieved_documents=reranked_documents
        )

        # --------------------------------------------------
        # 5. Generate diagnostic response
        # --------------------------------------------------

        diagnostic_result = self.generator.generate(prompts)

        # --------------------------------------------------
        # 6. Preserve normal production behaviour
        # --------------------------------------------------

        if not return_trace:
            return diagnostic_result

        # --------------------------------------------------
        # 7. Extract retrieved document contents for evaluation
        # --------------------------------------------------

        retrieved_contexts = []
        retrieved_document_details = []

        for rank, document in enumerate(reranked_documents, start=1):

            # Handle LangChain Document objects
            if isinstance(document, dict):

                content = (
                    document.get("page_content")
                    or document.get("text")
                    or document.get("content")
                    or document.get("document")
                )

                metadata = document.get("metadata", {})

            else:

                content = getattr(
                    document,
                    "page_content",
                    None
                )

                metadata = getattr(
                    document,
                    "metadata",
                    {}

                )

            # Handle plain text documents
            if content is None and isinstance(document, str):
                content = document

            if not isinstance(content, str) or not content.strip():
                raise ValueError(
                    f"Retrieved document at rank {rank} "
                    "does not contain usable text."
                )

            content = content.strip()

            retrieved_contexts.append(content)

            retrieved_document_details.append({
                "rank": rank,
                "page_content": content,
                "metadata": metadata or {}
            })

        # --------------------------------------------------
        # 8. Return complete evaluation trace
        # --------------------------------------------------

        return {
            "record_id": (
                f"{telemetry.get('asset_id', 'unknown')}_"
                f"{telemetry.get('timestamp_id', 'unknown')}"
            ),

            "asset_id": telemetry.get("asset_id"),

            "asset_type": telemetry.get("asset_type"),

            "timestamp_id": telemetry.get("timestamp_id"),

            "anomaly_count": telemetry.get("anomaly_count"),

            "anomaly_features": telemetry.get("anomaly_features"),

            "is_anomaly": telemetry.get("is_anomaly"),

            "user_input": diagnostic_query,

            "diagnostic_query": diagnostic_query,

            "diagnostic_context": diagnostic_context,

            "retrieved_contexts": retrieved_contexts,

            "retrieved_documents": retrieved_document_details,

            "response": diagnostic_result,

            "retrieval_statistics": {
                "initial_documents_retrieved": len(retrieved_documents),
                "documents_after_reranking": len(reranked_documents)
            }
        }