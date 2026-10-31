

from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# ==================================================
# 1. CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DIR = PROJECT_ROOT / "processed_documents" / "ChromaDB"

COLLECTION_NAME = "greenko_documents"

EMBEDDING_MODEL_NAME = "BAAI/bge-small-en-v1.5"


# ==================================================
# 2. INITIALIZE RETRIEVER
# ==================================================

class DenseRetriever:

    def __init__(self):

        # Load embedding model
        self.model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

        # Connect to existing persistent ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        print("Dense retriever initialized.")
        print("Collection:", COLLECTION_NAME)
        print("Stored chunks:", self.collection.count())


    # ==================================================
    # 3. RETRIEVE RELEVANT CHUNKS
    # ==================================================

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        asset_type: str | None = None
    ):

        if not query.strip():
            raise ValueError("Query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        # Generate query embedding
        query_embedding = self.model.encode(
            query,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Optional asset-type filtering
        where_filter = (
            {"asset_type": asset_type}
            if asset_type
            else None
        )

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(top_k, self.collection.count()),
            where=where_filter,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        # Format results
        retrieved_chunks = []

        for i in range(len(results["ids"][0])):

            distance = results["distances"][0][i]

            retrieved_chunks.append({
                "chunk_id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "cosine_distance": distance,
                "similarity_score": 1 - distance
            })

        return retrieved_chunks