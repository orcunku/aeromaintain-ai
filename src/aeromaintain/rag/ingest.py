"""
AeroMaintain AI - RAG Document Ingestion

Loads synthetic maintenance Markdown documents and converts them
into metadata-aware chunks for retrieval.

The ingestion layer is intentionally independent of a specific
embedding model or vector database so that retrieval backends can
be changed later without rewriting document parsing logic.

Synthetic maintenance documentation only.
"""

from dataclasses import dataclass
from pathlib import Path
import re

from aeromaintain.config import DOCUMENTS_DIR


@dataclass
class DocumentChunk:
    """A retrieval-ready chunk of a maintenance document."""

    chunk_id: str
    document_id: str
    title: str
    component_type: str
    section: str
    content: str
    source_file: str


def read_markdown_documents(
    documents_directory: Path = DOCUMENTS_DIR,
) -> list[Path]:
    """Return all Markdown knowledge-base documents."""

    document_paths = sorted(
        documents_directory.glob("*.md")
    )

    if not document_paths:
        raise FileNotFoundError(
            f"No Markdown documents found in "
            f"{documents_directory}"
        )

    return document_paths


def extract_metadata(
    text: str,
) -> tuple[str, str, str]:
    """Extract title, document ID, and component type."""

    title_match = re.search(
        r"^#\s+(.+)$",
        text,
        flags=re.MULTILINE,
    )

    document_id_match = re.search(
        r"^Document ID:\s*(.+)$",
        text,
        flags=re.MULTILINE,
    )

    component_type_match = re.search(
        r"^Component Type:\s*(.+)$",
        text,
        flags=re.MULTILINE,
    )

    if not all(
        [
            title_match,
            document_id_match,
            component_type_match,
        ]
    ):
        raise ValueError(
            "Document is missing required metadata."
        )

    return (
        title_match.group(1).strip(),
        document_id_match.group(1).strip(),
        component_type_match.group(1).strip(),
    )


def split_markdown_sections(
    text: str,
) -> list[tuple[str, str]]:
    """
    Split a Markdown document by level-two headings.

    Each section becomes a natural retrieval unit.
    """

    section_pattern = re.compile(
        r"^##\s+(.+)$",
        flags=re.MULTILINE,
    )

    matches = list(
        section_pattern.finditer(text)
    )

    sections = []

    for index, match in enumerate(matches):
        section_name = (
            match.group(1).strip()
        )

        content_start = match.end()

        if index + 1 < len(matches):
            content_end = (
                matches[index + 1].start()
            )
        else:
            content_end = len(text)

        section_content = (
            text[
                content_start:content_end
            ]
            .strip()
        )

        if section_content:
            sections.append(
                (
                    section_name,
                    section_content,
                )
            )

    return sections


def chunk_document(
    document_path: Path,
) -> list[DocumentChunk]:
    """Convert one Markdown document into retrieval chunks."""

    text = document_path.read_text(
        encoding="utf-8"
    )

    (
        title,
        document_id,
        component_type,
    ) = extract_metadata(text)

    sections = split_markdown_sections(
        text
    )

    chunks = []

    for index, (
        section_name,
        section_content,
    ) in enumerate(
        sections,
        start=1,
    ):
        chunk_id = (
            f"{document_id}-CHUNK-{index:02d}"
        )

        chunk_content = (
            f"Title: {title}\n"
            f"Component Type: {component_type}\n"
            f"Section: {section_name}\n\n"
            f"{section_content}"
        )

        chunks.append(
            DocumentChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                title=title,
                component_type=component_type,
                section=section_name,
                content=chunk_content,
                source_file=document_path.name,
            )
        )

    return chunks


def load_knowledge_base(
    documents_directory: Path = DOCUMENTS_DIR,
) -> list[DocumentChunk]:
    """Load and chunk the complete maintenance knowledge base."""

    document_paths = (
        read_markdown_documents(
            documents_directory
        )
    )

    all_chunks = []

    for document_path in document_paths:
        document_chunks = (
            chunk_document(
                document_path
            )
        )

        all_chunks.extend(
            document_chunks
        )

    return all_chunks


def main() -> None:
    chunks = load_knowledge_base()

    document_count = len(
        {
            chunk.document_id
            for chunk in chunks
        }
    )

    print(
        f"Loaded documents: {document_count}"
    )

    print(
        f"Generated chunks: {len(chunks)}"
    )

    print(
        "\nChunks by document:"
    )

    document_ids = sorted(
        {
            chunk.document_id
            for chunk in chunks
        }
    )

    for document_id in document_ids:
        count = sum(
            chunk.document_id
            == document_id
            for chunk in chunks
        )

        print(
            f"  {document_id}: {count}"
        )

    print(
        "\nExample chunk:"
    )

    example = chunks[0]

    print(
        f"Chunk ID: {example.chunk_id}"
    )
    print(
        f"Component: {example.component_type}"
    )
    print(
        f"Section: {example.section}"
    )
    print()
    print(
        example.content
    )


if __name__ == "__main__":
    main()