
import json
from pathlib import Path

from llama_index.core import Document
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter
)

# 1. PROJECT PATHS

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MARKDOWN_DIR = (
    PROJECT_ROOT / "processed_documents" / "Docling" / "Markdown"
)

CHUNKS_DIR = (
    PROJECT_ROOT / "processed_documents" / "Chunks"
)

CHUNKS_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. CHUNKING CONFIGURATION
# ==================================================

CHUNK_SIZE = 1000
MAX_CHUNK_SIZE = 1200
CHUNK_OVERLAP = 100


# ==================================================
# 3. ASSET TYPE CLASSIFICATION
# ==================================================


def extract_asset_type(filename):

    filename = filename.lower()

    # Normalize underscores and spaces
    normalized_name = filename.replace("_", "-").replace(" ", "-")

    # Wind documents
    if (
        "1024891" in normalized_name
        or "wind-turbine-maintenance-guide" in normalized_name
        or "wind-turbine-fault-diagnosis-and-predictive-maintenance" in normalized_name
    ):
        return "Wind"

    # Solar documents
    elif (
        "applsci-15-11461" in normalized_name
        or "pv-failure-monitoring" in normalized_name
    ):
        return "Solar"

    else:
        raise ValueError(
            f"Unable to identify asset type: {filename}"
        )


# ==================================================
# 4. DOCUMENT METADATA EXTRACTION
# ==================================================

def extract_document_metadata(markdown_path):

    document_id = markdown_path.stem

    source_file = document_id + ".pdf"

    asset_type = extract_asset_type(source_file)

    return {
        "document_id": document_id,
        "source_file": source_file,
        "asset_type": asset_type
    }


# ==================================================
# 5. SECTION METADATA EXTRACTION
# ==================================================

def extract_section_metadata(node):

    heading_path = node.metadata.get("header_path", "")

    if isinstance(heading_path, list):
        heading_path = " > ".join(heading_path)

    if not isinstance(heading_path, str):
        heading_path = str(heading_path)

    section_title = (
        heading_path.split(" > ")[-1]
        if heading_path
        else "General"
    )

    return {
        "section_title": section_title,
        "heading_path": heading_path
    }


# ==================================================
# 6. STRUCTURE-AWARE CHUNKING
# ==================================================

def chunk_document(markdown_path):

    # Read Markdown
    markdown_text = markdown_path.read_text(
        encoding="utf-8"
    )

    if not markdown_text.strip():
        raise ValueError(
            f"Empty Markdown document: {markdown_path.name}"
        )

    # Extract document metadata
    document_metadata = extract_document_metadata(
        markdown_path
    )

    # Create LlamaIndex Document
    document = Document(
        text=markdown_text,
        metadata=document_metadata
    )

    # First stage: Structure-aware parsing
    markdown_parser = MarkdownNodeParser.from_defaults()

    section_nodes = markdown_parser.get_nodes_from_documents(
        [document]
    )

    # Second stage: Split oversized sections
    sentence_splitter = SentenceSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )

    final_chunks = []

    for node in section_nodes:

        section_metadata = extract_section_metadata(node)

        # Split section according to token limits
        text_parts = sentence_splitter.split_text(node.text)

        for text_part in text_parts:

            if not text_part.strip():
                continue

            chunk_index = len(final_chunks)

            chunk = {
                "chunk_id": (
                    f"{document_metadata['document_id']}"
                    f"_chunk_{chunk_index + 1:03d}"
                ),

                "chunk_index": chunk_index,

                "text": text_part,

                "metadata": {
                    **document_metadata,
                    **section_metadata
                }
            }

            final_chunks.append(chunk)

    return final_chunks


# ==================================================
# 7. SAVE CHUNKS AS JSON
# ==================================================

def save_chunks(chunks, markdown_path):

    output_path = (
        CHUNKS_DIR / f"{markdown_path.stem}.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:

        json.dump(
            chunks,
            file,
            indent=4,
            ensure_ascii=False
        )

    return output_path


# ==================================================
# 8. PROCESS ALL DOCUMENTS
# ==================================================

def process_all_documents():

    markdown_files = sorted(
        MARKDOWN_DIR.glob("*.md")
    )

    if not markdown_files:
        raise FileNotFoundError(
            f"No Markdown files found in: {MARKDOWN_DIR}"
        )

    total_chunks = 0

    for markdown_path in markdown_files:

        print("\n" + "=" * 55)
        print(f"Processing: {markdown_path.name}")
        print("=" * 55)

        chunks = chunk_document(markdown_path)

        if not chunks:
            print("Warning: No chunks generated.")
            continue

        output_path = save_chunks(
            chunks,
            markdown_path
        )

        asset_type = chunks[0]["metadata"]["asset_type"]

        print(f"Asset Type: {asset_type}")
        print(f"Total Chunks: {len(chunks)}")
        print(f"Saved: {output_path}")

        total_chunks += len(chunks)

    print("\n" + "=" * 55)
    print("CHUNKING COMPLETED")
    print(f"Documents Processed: {len(markdown_files)}")
    print(f"Total Chunks Generated: {total_chunks}")
    print(f"Output Directory: {CHUNKS_DIR}")
    print("=" * 55)


# ==================================================
# 9. SCRIPT EXECUTION
# ==================================================

if __name__ == "__main__":

    process_all_documents()
