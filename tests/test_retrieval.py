from aeromaintain.rag.ingest import load_knowledge_base


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

    assert len(chunk_ids) == len(
        set(chunk_ids)
    )


def test_expected_component_types_are_present():
    chunks = load_knowledge_base()

    component_types = {
        chunk.component_type
        for chunk in chunks
    }

    assert component_types == {
        "AIR_CYCLE_MACHINE",
        "ACTUATOR",
        "FUEL_PUMP",
        "GENERATOR",
        "HYDRAULIC_PUMP",
    }


def test_each_document_has_six_sections():
    chunks = load_knowledge_base()

    document_ids = {
        chunk.document_id
        for chunk in chunks
    }

    for document_id in document_ids:
        document_chunks = [
            chunk
            for chunk in chunks
            if chunk.document_id
            == document_id
        ]

        assert len(document_chunks) == 6