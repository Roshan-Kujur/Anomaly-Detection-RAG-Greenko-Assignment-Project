
from pathlib import Path

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import (
    PdfPipelineOptions,
    TableFormerMode,
)
from docling.document_converter import (
    DocumentConverter,
    PdfFormatOption,
)
from docling_core.types.doc import ImageRefMode


# Project paths
PROJECT_ROOT = Path(__file__).resolve().parents[2]

PDF_DIRECTORY = PROJECT_ROOT / "PDFs for RAG"

OUTPUT_DIRECTORY = PROJECT_ROOT / "processed_documents" / "docling"

MARKDOWN_DIRECTORY = OUTPUT_DIRECTORY / "markdown"
JSON_DIRECTORY = OUTPUT_DIRECTORY / "json"
TABLE_DIRECTORY = OUTPUT_DIRECTORY / "tables"


# Create output directories
for directory in [
    MARKDOWN_DIRECTORY,
    JSON_DIRECTORY,
    TABLE_DIRECTORY,
]:
    directory.mkdir(parents=True, exist_ok=True)


# Docling configuration
pipeline_options = PdfPipelineOptions()

pipeline_options.do_ocr = False
pipeline_options.do_table_structure = True
pipeline_options.table_structure_options.mode = TableFormerMode.FAST
pipeline_options.generate_picture_images = True
pipeline_options.generate_page_images = False
pipeline_options.images_scale = 1.0


# Initialize converter once
converter = DocumentConverter(
    allowed_formats=[InputFormat.PDF],
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=pipeline_options
        )
    },
)


# Process all PDFs
def process_all_pdfs():

    results = []

    pdf_files = sorted(PDF_DIRECTORY.glob("*.pdf"))

    for pdf_path in pdf_files:

        filename = pdf_path.stem

        document = converter.convert(pdf_path).document

        # Export Markdown
        document.save_as_markdown(
            MARKDOWN_DIRECTORY / f"{filename}.md",
            image_mode=ImageRefMode.REFERENCED,
        )

        # Export JSON
        document.save_as_json(
            JSON_DIRECTORY / f"{filename}.json"
        )

        # Extract tables
        for index, table in enumerate(
            document.tables,
            start=1
        ):

            dataframe = table.export_to_dataframe(
                doc=document
            )

            dataframe.to_csv(
                TABLE_DIRECTORY / f"{filename}_table_{index}.csv",
                index=False,
                encoding="utf-8-sig",
            )

        results.append({
            "source_file": pdf_path.name,
            "pages": len(document.pages),
            "tables": len(document.tables),
            "pictures": len(document.pictures),
        })

    return results
