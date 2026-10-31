
from pathlib import Path
from typing import List


# Supported document formats
SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}


def load_document_paths(
    source_dir: str,
    recursive: bool = True
) -> List[Path]:
    """
    Discover supported documents from a directory.

    Args:
        source_dir: Directory containing knowledge-base documents.
        recursive: Whether to search subdirectories.

    Returns:
        List of document file paths.
    """

    source_path = Path(source_dir)

    # Validate directory
    if not source_path.exists():
        raise FileNotFoundError(
            f"Knowledge base directory not found: {source_path}"
        )

    if not source_path.is_dir():
        raise NotADirectoryError(
            f"Expected a directory, received: {source_path}"
        )

    # Discover documents
    if recursive:
        files = source_path.rglob("*")
    else:
        files = source_path.glob("*")

    document_paths = [
        file_path
        for file_path in files
        if file_path.is_file()
        and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]

    if not document_paths:
        print("No supported documents found.")

    else:
        print(f"Found {len(document_paths)} documents.")

    return sorted(document_paths)
