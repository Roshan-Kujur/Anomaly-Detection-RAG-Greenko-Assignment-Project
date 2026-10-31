
import json
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# -----------------------------------------
# PATH CONFIGURATION
# -----------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_DIR = PROJECT_ROOT / "processed_documents" / "Chunks"

CHROMA_DIR = PROJECT_ROOT / "processed_documents" / "ChromaDB"


# -----------------------------------------
# MODEL CONFIGURATION
# -----------------------------------------

MODEL_NAME = "BAAI/bge-small-en-v1.5"

COLLECTION_NAME = "greenko_documents"


# -----------------------------------------
# LOAD EMBEDDING MODEL
# -----------------------------------------

def load_embedding_model():

    model = SentenceTransformer(MODEL_NAME)

    print("Embedding model loaded successfully.")

    return model


# -----------------------------------------
# INITIALIZE PERSISTENT CHROMADB
# -----------------------------------------

def initialize_chromadb():

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )

    print("Persistent ChromaDB initialized.")

    return collection


# -----------------------------------------
# LOAD CHUNKS FROM JSON FILES
# -----------------------------------------

def load_chunks():

    all_chunks = []

    json_files = sorted(CHUNKS_DIR.glob("*.json"))

    if not json_files:
        raise FileNotFoundError(
            f"No JSON chunk files found in {CHUNKS_DIR}"
        )

    for file_path in json_files:

        with open(file_path, "r", encoding="utf-8") as file:
            chunks = json.load(file)

        all_chunks.extend(chunks)

    print(f"Total chunks loaded: {len(all_chunks)}")

    return all_chunks


# -----------------------------------------
# GENERATE EMBEDDINGS AND STORE
# -----------------------------------------

def generate_and_store_embeddings():

    model = load_embedding_model()

    collection = initialize_chromadb()

    chunks = load_chunks()

    batch_size = 16

    for start in range(0, len(chunks), batch_size):

        batch = chunks[start:start + batch_size]

        texts = [
            chunk["text"]
            for chunk in batch
        ]

        # Generate embeddings
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        # Prepare IDs and metadata
        ids = []
        metadatas = []

        for chunk in batch:

            metadata = chunk.get("metadata", {})

            document_id = metadata.get(
                "document_id", "unknown"
            )

            unique_id = (
                f"{document_id}_{chunk['chunk_id']}"
            )

            ids.append(unique_id)

            # ChromaDB metadata supports simple values
            clean_metadata = {
                key: value
                for key, value in metadata.items()
                if isinstance(value, (str, int, float, bool))
            }

            metadatas.append(clean_metadata)

        # Store in ChromaDB
        collection.upsert(
            ids=ids,
            embeddings=embeddings.tolist(),
            documents=texts,
            metadatas=metadatas
        )

        print(
            f"Stored {min(start + len(batch), len(chunks))}"
            f" / {len(chunks)} chunks"
        )

    print("\nEmbedding generation completed.")
    print("Vectors stored successfully in ChromaDB.")
    print("Total vectors:", collection.count())
    print("Database location:", CHROMA_DIR)

    return collection


# -----------------------------------------
# EXECUTION
# -----------------------------------------

if __name__ == "__main__":

    generate_and_store_embeddings()
