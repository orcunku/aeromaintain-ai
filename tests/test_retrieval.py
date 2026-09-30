from aeromaintain.rag.ingest import load_knowledge_base
from aeromaintain.rag.retrieve import (
    LSAMaintenanceRetriever,
    MaintenanceRetriever,
)


EXPECTED_COMPONENT_TYPES = {
    "HYDRAULIC_PUMP",
    "GENERATOR",
    "AIR_CYCLE_MACHINE",
    "FUEL_PUMP",
    "ACTUATOR",
}


def test_knowledge_base_document_count():
    chunks = load_knowledge_base()

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    assert len(document_ids) == 5


def test_knowledge_base_chunk_count():
    chunks = load_knowledge_base()

    assert len(chunks) == 30


def test_chunk_metadata_is_present():
    chunks = load_knowledge_base()

    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.document_id
        assert chunk.title
        assert chunk.component_type
        assert chunk.section
        assert chunk.content
        assert chunk.source_file


def test_chunk_ids_are_unique():
    chunks = load_knowledge_base()

    chunk_ids = [
        chunk.chunk_id
        for chunk in chunks
    ]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_expected_component_types_are_present():
    chunks = load_knowledge_base()

    component_types = {
        chunk.component_type
        for chunk in chunks
    }

    assert component_types == EXPECTED_COMPONENT_TYPES


def test_each_document_has_six_sections():
    chunks = load_knowledge_base()

    section_counts = {}

    for chunk in chunks:
        section_counts.setdefault(
            chunk.document_id,
            0,
        )
        section_counts[chunk.document_id] += 1

    assert len(section_counts) == 5

    for count in section_counts.values():
        assert count == 6


def test_retriever_ranks_relevant_component_first():
    retriever = MaintenanceRetriever()

    results = retriever.retrieve(
        query=(
            "hydraulic pump vibration is increasing "
            "and temperature is elevated"
        ),
        top_k=5,
    )

    assert len(results) == 5
    assert results[0].chunk.component_type == "HYDRAULIC_PUMP"
    assert results[0].chunk.section == "Typical Condition Indicators"
    assert results[0].score > 0


def test_lsa_retriever_returns_requested_number_of_results():
    retriever = LSAMaintenanceRetriever(
        n_components=10,
    )

    results = retriever.retrieve(
        query=(
            "hydraulic pump vibration is increasing "
            "and temperature is elevated"
        ),
        top_k=5,
    )

    assert len(results) == 5


def test_lsa_retriever_ranks_relevant_component_first():
    retriever = LSAMaintenanceRetriever(
        n_components=10,
    )

    results = retriever.retrieve(
        query=(
            "hydraulic pump vibration is increasing "
            "and temperature is elevated"
        ),
        top_k=5,
    )

    assert results[0].chunk.component_type == "HYDRAULIC_PUMP"
    assert results[0].chunk.section == "Typical Condition Indicators"
    assert results[0].score > 0


def test_lsa_component_filter_returns_only_requested_component():
    retriever = LSAMaintenanceRetriever(
        n_components=10,
    )

    results = retriever.retrieve(
        query="increasing vibration and elevated temperature",
        top_k=5,
        component_type="HYDRAULIC_PUMP",
    )

    assert len(results) == 5

    assert all(
        result.chunk.component_type == "HYDRAULIC_PUMP"
        for result in results
    )