
import re
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi


# 1. CONFIGURATION


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHROMA_DIR = PROJECT_ROOT / "processed_documents" / "ChromaDB"

COLLECTION_NAME = "greenko_documents"


# 2. TEXT TOKENIZATION


def tokenize_text(text: str) -> list[str]:
    """
    Convert text into lowercase tokens for BM25.
    """

    tokens = re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())

    return tokens



# 3. BM25 RETRIEVER


class BM25Retriever:

    def __init__(self):

        # Connect to existing ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

        # Load existing documents and metadata
        stored_data = self.collection.get(
            include=["documents", "metadatas"]
        )

        self.documents = stored_data["documents"]
        self.metadatas = stored_data["metadatas"]
        self.chunk_ids = stored_data["ids"]

        if not self.documents:
            raise ValueError("No documents found in ChromaDB.")

        # Tokenize all documents
        self.tokenized_corpus = [
            tokenize_text(document)
            for document in self.documents
        ]

        # Build BM25 index
        self.bm25 = BM25Okapi(
            self.tokenized_corpus
        )

        print("BM25 retriever initialized.")
        print("Collection:", COLLECTION_NAME)
        print("Indexed chunks:", len(self.documents))


    # 4. RETRIEVE RELEVANT CHUNKS
  

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

        # Tokenize query
        tokenized_query = tokenize_text(query)

        if not tokenized_query:
            raise ValueError("Query contains no searchable terms.")

        # Calculate BM25 scores
        scores = self.bm25.get_scores(
            tokenized_query
        )

        # Apply optional asset-type filtering
        candidate_indices = []

        for index, metadata in enumerate(self.metadatas):

            if (
                asset_type is None
                or metadata.get("asset_type") == asset_type
            ):
                candidate_indices.append(index)

        if not candidate_indices:
            return []

        # Rank filtered documents by BM25 score
        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: scores[index],
            reverse=True
        )

        # Select top-k results
        top_indices = ranked_indices[:top_k]

        retrieved_chunks = []

        for index in top_indices:

            retrieved_chunks.append({

                "chunk_id": self.chunk_ids[index],

                "text": self.documents[index],

                "metadata": self.metadatas[index],

                "bm25_score": float(scores[index])

            })

        return retrieved_chunks
