
from backend.rag_system.retriever import DenseRetriever
from backend.rag_system.bm25_retriever import BM25Retriever


# ==================================================
# 1. HYBRID RETRIEVER
# ==================================================

class HybridRetriever:

    def __init__(
        self,
        rrf_k: int = 60,
        candidate_k: int = 20
    ):

        self.dense_retriever = DenseRetriever()
        self.bm25_retriever = BM25Retriever()

        self.rrf_k = rrf_k
        self.candidate_k = candidate_k

        print("Hybrid retriever initialized.")
        print("RRF constant:", self.rrf_k)
        print("Candidate chunks per retriever:", self.candidate_k)


    # ==================================================
    # 2. RECIPROCAL RANK FUSION
    # ==================================================

    def reciprocal_rank_fusion(
        self,
        dense_results,
        bm25_results
    ):

        fused_results = {}

        # Process dense retrieval rankings
        for rank, result in enumerate(dense_results, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id not in fused_results:

                fused_results[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None
                }

            fused_results[chunk_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

            fused_results[chunk_id]["dense_rank"] = rank


        # Process BM25 retrieval rankings
        for rank, result in enumerate(bm25_results, start=1):

            chunk_id = result["chunk_id"]

            if chunk_id not in fused_results:

                fused_results[chunk_id] = {
                    **result,
                    "rrf_score": 0.0,
                    "dense_rank": None,
                    "bm25_rank": None
                }

            fused_results[chunk_id]["rrf_score"] += (
                1 / (self.rrf_k + rank)
            )

            fused_results[chunk_id]["bm25_rank"] = rank


        # Sort by combined RRF score
        ranked_results = sorted(
            fused_results.values(),
            key=lambda x: x["rrf_score"],
            reverse=True
        )

        return ranked_results


    # ==================================================
    # 3. HYBRID RETRIEVAL
    # ==================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        asset_type: str | None = None
    ):

        if not query or not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Dense semantic retrieval
        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=self.candidate_k,
            asset_type=asset_type
        )

        # BM25 keyword retrieval
        bm25_results = self.bm25_retriever.retrieve(
            query=query,
            top_k=self.candidate_k,
            asset_type=asset_type
        )

        # Combine both retrieval rankings
        hybrid_results = self.reciprocal_rank_fusion(
            dense_results=dense_results,
            bm25_results=bm25_results
        )

        return hybrid_results[:top_k]
