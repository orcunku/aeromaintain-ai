from dataclasses import dataclass

from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from aeromaintain.rag.ingest import DocumentChunk, load_knowledge_base


@dataclass
class RetrievalResult:
    """A single retrieval result with its similarity score."""

    chunk: DocumentChunk
    score: float


class MaintenanceRetriever:
    """
    Lexical maintenance-document retriever using TF-IDF.

    This class is intentionally kept as the baseline retrieval system so that
    it can later be compared against the LSA-based semantic retriever.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
    ):
        if chunks is None:
            chunks = load_knowledge_base()

        if not chunks:
            raise ValueError("At least one document chunk is required.")

        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )

        corpus = [chunk.content for chunk in self.chunks]

        self.chunk_matrix = self.vectorizer.fit_transform(corpus)

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        component_type: str | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_vector = self.vectorizer.transform([query])

        similarities = cosine_similarity(
            query_vector,
            self.chunk_matrix,
        ).flatten()

        candidate_indices = []

        for index, chunk in enumerate(self.chunks):
            if (
                component_type is not None
                and chunk.component_type != component_type
            ):
                continue

            candidate_indices.append(index)

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: similarities[index],
            reverse=True,
        )

        selected_indices = ranked_indices[:top_k]

        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(similarities[index]),
            )
            for index in selected_indices
        ]


class LSAMaintenanceRetriever:
    """
    Semantic maintenance-document retriever using Latent Semantic Analysis.

    The retrieval pipeline is:

        text
          -> TF-IDF
          -> TruncatedSVD
          -> latent semantic representation
          -> cosine similarity

    This implementation is fully local and does not require an external
    embedding API or a downloaded transformer model.
    """

    def __init__(
        self,
        chunks: list[DocumentChunk] | None = None,
        n_components: int = 10,
    ):
        if chunks is None:
            chunks = load_knowledge_base()

        if not chunks:
            raise ValueError("At least one document chunk is required.")

        if n_components < 1:
            raise ValueError("n_components must be at least 1.")

        self.chunks = chunks

        self.vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )

        corpus = [chunk.content for chunk in self.chunks]

        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)

        # TruncatedSVD cannot use more useful latent dimensions than the
        # available matrix dimensions. Keeping this bounded also makes the
        # retriever safer when tested with a smaller custom chunk collection.
        max_components = min(
            self.tfidf_matrix.shape[0] - 1,
            self.tfidf_matrix.shape[1] - 1,
        )

        if max_components < 1:
            raise ValueError(
                "The document collection is too small for LSA retrieval."
            )

        self.n_components = min(
            n_components,
            max_components,
        )

        self.svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=42,
        )

        self.chunk_embeddings = self.svd.fit_transform(
            self.tfidf_matrix
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        component_type: str | None = None,
    ) -> list[RetrievalResult]:
        if not query.strip():
            raise ValueError("Query must not be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_tfidf = self.vectorizer.transform([query])

        query_embedding = self.svd.transform(
            query_tfidf
        )

        similarities = cosine_similarity(
            query_embedding,
            self.chunk_embeddings,
        ).flatten()

        candidate_indices = []

        for index, chunk in enumerate(self.chunks):
            if (
                component_type is not None
                and chunk.component_type != component_type
            ):
                continue

            candidate_indices.append(index)

        ranked_indices = sorted(
            candidate_indices,
            key=lambda index: similarities[index],
            reverse=True,
        )

        selected_indices = ranked_indices[:top_k]

        return [
            RetrievalResult(
                chunk=self.chunks[index],
                score=float(similarities[index]),
            )
            for index in selected_indices
        ]


if __name__ == "__main__":
    query = (
        "hydraulic pump vibration is increasing "
        "and temperature is elevated"
    )

    print("\nTF-IDF RETRIEVAL")
    print("=" * 70)

    tfidf_retriever = MaintenanceRetriever()

    tfidf_results = tfidf_retriever.retrieve(
        query=query,
        top_k=5,
    )

    for rank, result in enumerate(tfidf_results, start=1):
        print(
            f"{rank}. "
            f"{result.chunk.component_type} | "
            f"{result.chunk.section} | "
            f"score={result.score:.4f}"
        )

    print("\nLSA SEMANTIC RETRIEVAL")
    print("=" * 70)

    lsa_retriever = LSAMaintenanceRetriever(
        n_components=10,
    )

    lsa_results = lsa_retriever.retrieve(
        query=query,
        top_k=5,
    )

    for rank, result in enumerate(lsa_results, start=1):
        print(
            f"{rank}. "
            f"{result.chunk.component_type} | "
            f"{result.chunk.section} | "
            f"score={result.score:.4f}"
        )