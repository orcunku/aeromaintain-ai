"""
AeroMaintain AI - RAG Retrieval

Provides a lightweight lexical retrieval baseline over the
synthetic maintenance knowledge base.

The baseline uses:
- TF-IDF document representations
- Cosine similarity
- Optional component-type filtering

This creates an interpretable retrieval baseline that can later
be compared with embedding-based semantic retrieval.

Synthetic maintenance documentation only.
"""

from dataclasses import dataclass

from sklearn.feature_extraction.text import (
    TfidfVectorizer,
)
from sklearn.metrics.pairwise import (
    cosine_similarity,
)

from aeromaintain.rag.ingest import (
    DocumentChunk,
    load_knowledge_base,
)


@dataclass
class RetrievalResult:
    """One ranked retrieval result."""

    chunk: DocumentChunk
    score: float


class MaintenanceRetriever:
    """TF-IDF retrieval over maintenance knowledge-base chunks."""

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
    ) -> None:
        if chunks is None:
            chunks = load_knowledge_base()

        if not chunks:
            raise ValueError(
                "At least one document chunk is required."
            )

        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )

        corpus = [
            chunk.content
            for chunk in self.chunks
        ]

        self.chunk_matrix = (
            self.vectorizer.fit_transform(
                corpus
            )
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        component_type: str | None = None,
    ) -> list[RetrievalResult]:
        """
        Retrieve the most relevant knowledge-base chunks.

        When component_type is provided, only chunks belonging
        to that component family are eligible for retrieval.
        """

        if not query.strip():
            raise ValueError(
                "Query must not be empty."
            )

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1."
            )

        query_vector = (
            self.vectorizer.transform(
                [query]
            )
        )

        similarities = (
            cosine_similarity(
                query_vector,
                self.chunk_matrix,
            )
            .flatten()
        )

        candidate_indices = []

        for index, chunk in enumerate(
            self.chunks
        ):
            if (
                component_type is not None
                and chunk.component_type
                != component_type
            ):
                continue

            candidate_indices.append(
                index
            )

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: similarities[index],
            reverse=True,
        )

        selected_indices = (
            ranked_indices[:top_k]
        )

        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(
                    similarities[index]
                ),
            )
            for index in selected_indices
        ]


def main() -> None:
    retriever = MaintenanceRetriever()

    query = (
        "hydraulic pump vibration is increasing "
        "and temperature is elevated"
    )

    results = retriever.retrieve(
        query=query,
        top_k=5,
    )

    print(
        f"Query: {query}"
    )

    print(
        "\nTop retrieval results:"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):
        print(
            f"\n{rank}. "
            f"{result.chunk.chunk_id}"
        )

        print(
            "Component:",
            result.chunk.component_type,
        )

        print(
            "Section:",
            result.chunk.section,
        )

        print(
            "Score:",
            f"{result.score:.4f}",
        )


if __name__ == "__main__":
    main()