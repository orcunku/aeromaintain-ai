import json
from pathlib import Path

from aeromaintain.rag.retrieve import (
    LSAMaintenanceRetriever,
    MaintenanceRetriever,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_FILE = PROJECT_ROOT / "evaluation" / "retrieval_queries.json"


def load_evaluation_queries() -> list[dict]:
    """Load the retrieval ground-truth query set."""
    with QUERY_FILE.open("r", encoding="utf-8") as file:
        return json.load(file)


def is_relevant(result, query_item: dict) -> bool:
    """
    A retrieved chunk is relevant only when both the component type
    and document section match the ground truth.
    """
    return (
        result.chunk.component_type
        == query_item["expected_component_type"]
        and result.chunk.section
        == query_item["expected_section"]
    )


def reciprocal_rank(results, query_item: dict) -> float:
    """Return reciprocal rank of the first relevant result."""
    for rank, result in enumerate(results, start=1):
        if is_relevant(result, query_item):
            return 1.0 / rank

    return 0.0


def evaluate_retriever(
    retriever,
    queries: list[dict],
    top_k: int = 5,
) -> dict:
    """Evaluate one retriever against the ground-truth query set."""
    hit_at_1 = 0
    hit_at_3 = 0
    reciprocal_ranks = []
    query_results = []

    for query_item in queries:
        results = retriever.retrieve(
            query=query_item["query"],
            top_k=top_k,
        )

        rank = None

        for index, result in enumerate(results, start=1):
            if is_relevant(result, query_item):
                rank = index
                break

        if rank == 1:
            hit_at_1 += 1

        if rank is not None and rank <= 3:
            hit_at_3 += 1

        rr = reciprocal_rank(
            results,
            query_item,
        )
        reciprocal_ranks.append(rr)

        top_result = results[0]

        query_results.append(
            {
                "query_id": query_item["query_id"],
                "query": query_item["query"],
                "expected_component_type": (
                    query_item["expected_component_type"]
                ),
                "expected_section": (
                    query_item["expected_section"]
                ),
                "retrieved_component_type": (
                    top_result.chunk.component_type
                ),
                "retrieved_section": (
                    top_result.chunk.section
                ),
                "relevant_rank": rank,
                "top_score": top_result.score,
            }
        )

    query_count = len(queries)

    return {
        "hit_at_1": hit_at_1 / query_count,
        "hit_at_3": hit_at_3 / query_count,
        "mrr": sum(reciprocal_ranks) / query_count,
        "query_results": query_results,
    }


def print_evaluation(
    name: str,
    evaluation: dict,
) -> None:
    """Print summary metrics and per-query retrieval results."""
    print("\n" + "=" * 80)
    print(name)
    print("=" * 80)

    print(
        f"Hit@1: {evaluation['hit_at_1']:.3f}"
    )
    print(
        f"Hit@3: {evaluation['hit_at_3']:.3f}"
    )
    print(
        f"MRR:   {evaluation['mrr']:.3f}"
    )

    print("\nPer-query results:")

    for result in evaluation["query_results"]:
        rank = result["relevant_rank"]

        rank_display = (
            str(rank)
            if rank is not None
            else "NOT FOUND"
        )

        print(
            f"{result['query_id']} | "
            f"rank={rank_display} | "
            f"expected="
            f"{result['expected_component_type']}/"
            f"{result['expected_section']} | "
            f"top="
            f"{result['retrieved_component_type']}/"
            f"{result['retrieved_section']} | "
            f"score={result['top_score']:.4f}"
        )


def main() -> None:
    queries = load_evaluation_queries()

    tfidf_retriever = MaintenanceRetriever()

    lsa_retriever = LSAMaintenanceRetriever(
        n_components=10,
    )

    tfidf_evaluation = evaluate_retriever(
        retriever=tfidf_retriever,
        queries=queries,
        top_k=5,
    )

    lsa_evaluation = evaluate_retriever(
        retriever=lsa_retriever,
        queries=queries,
        top_k=5,
    )

    print_evaluation(
        name="TF-IDF RETRIEVAL EVALUATION",
        evaluation=tfidf_evaluation,
    )

    print_evaluation(
        name="LSA RETRIEVAL EVALUATION",
        evaluation=lsa_evaluation,
    )


if __name__ == "__main__":
    main()