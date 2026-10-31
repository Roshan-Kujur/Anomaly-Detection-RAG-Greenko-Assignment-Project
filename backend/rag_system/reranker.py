
from sentence_transformers import CrossEncoder


class CrossEncoderReranker:

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        max_length: int = 512
    ):
        self.model = CrossEncoder(
            model_name,
            max_length=max_length
        )

        print("Cross-Encoder reranker initialized.")
        print("Model:", model_name)

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5
    ) -> list[dict]:

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if not documents:
            return []

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        pairs = [
            (query, document["text"])
            for document in documents
        ]

        scores = self.model.predict(
            pairs,
            batch_size=16,
            show_progress_bar=False
        )

        reranked_results = []

        for document, score in zip(documents, scores):

            result = document.copy()
            result["rerank_score"] = float(score)

            reranked_results.append(result)

        reranked_results.sort(
            key=lambda x: x["rerank_score"],
            reverse=True
        )

        return reranked_results[:top_k]
