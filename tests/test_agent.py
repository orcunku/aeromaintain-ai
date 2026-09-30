from unittest.mock import patch

import pytest

from aeromaintain.agents.tools import MaintenanceTools


def test_search_maintenance_docs_returns_structured_evidence():
    """
    The document-search tool should return structured dictionaries
    that an agent or API can consume directly.
    """
    tools = MaintenanceTools()

    results = tools.search_maintenance_docs(
        query="hydraulic pump vibration increasing and temperature elevated",
        top_k=3,
    )

    assert len(results) == 3

    required_fields = {
        "score",
        "document_id",
        "title",
        "component_type",
        "section",
        "content",
        "source_file",
    }

    assert required_fields.issubset(
        results[0].keys()
    )


def test_search_maintenance_docs_ranks_hydraulic_evidence_first():
    """
    A hydraulic-pump condition query should rank the expected
    hydraulic condition-indicator chunk first.
    """
    tools = MaintenanceTools()

    results = tools.search_maintenance_docs(
        query="hydraulic pump vibration increasing and temperature elevated",
        top_k=3,
    )

    assert (
        results[0]["component_type"]
        == "HYDRAULIC_PUMP"
    )

    assert (
        results[0]["section"]
        == "Typical Condition Indicators"
    )


def test_search_maintenance_docs_respects_component_filter():
    """
    When a component filter is supplied, every returned result
    should belong to that component type.
    """
    tools = MaintenanceTools()

    results = tools.search_maintenance_docs(
        query="abnormal vibration and temperature",
        top_k=3,
        component_type="HYDRAULIC_PUMP",
    )

    assert len(results) == 3

    assert all(
        result["component_type"]
        == "HYDRAULIC_PUMP"
        for result in results
    )


def test_get_component_history_returns_expected_event():
    """
    The history tool should retrieve the known synthetic maintenance
    event for component CMP-00196.
    """
    tools = MaintenanceTools()

    history = tools.get_component_history(
        component_id="CMP-00196",
    )

    assert len(history) >= 1

    event = history[0]

    assert event["component_id"] == "CMP-00196"
    assert event["aircraft_id"] == "AC-040"
    assert event["component_type"] == "HYDRAULIC_PUMP"
    assert event["work_order_id"] == "WO-000002"
    assert event["event_type"] == "CORRECTIVE"
    assert event["fault_code"] == "FLT-761"
    assert event["corrective_action"] == "REPAIRED"


def test_get_component_history_returns_newest_first():
    """
    Maintenance history should be ordered from newest to oldest.
    """
    tools = MaintenanceTools()

    history = tools.get_component_history(
        component_id="CMP-00196",
        limit=10,
    )

    timestamps = [
        event["timestamp"]
        for event in history
    ]

    assert timestamps == sorted(
        timestamps,
        reverse=True,
    )


def test_get_component_history_unknown_component_returns_empty_list():
    """
    An unknown component should safely return an empty history
    instead of raising an unexpected error.
    """
    tools = MaintenanceTools()

    history = tools.get_component_history(
        component_id="CMP-DOES-NOT-EXIST",
    )

    assert history == []


def test_get_component_history_rejects_empty_component_id():
    """
    Empty component identifiers should be rejected.
    """
    tools = MaintenanceTools()

    with pytest.raises(
        ValueError,
        match="component_id must not be empty",
    ):
        tools.get_component_history(
            component_id="",
        )


def test_get_component_history_rejects_invalid_limit():
    """
    The history limit must be at least one.
    """
    tools = MaintenanceTools()

    with pytest.raises(
        ValueError,
        match="limit must be at least 1",
    ):
        tools.get_component_history(
            component_id="CMP-00196",
            limit=0,
        )


def test_calculate_failure_risk_returns_structured_result():
    """
    The predictive-risk tool should combine feature retrieval and
    model inference into a structured agent-facing result.

    The predictive model is mocked so this unit test does not depend
    on a local untracked joblib artifact.
    """
    fake_features = {
        "aircraft_type": "A320",
        "component_type": "HYDRAULIC_PUMP",
    }

    fake_prediction = {
        "risk_score": 0.71,
        "threshold": 0.72,
        "above_threshold": False,
        "interpretation": "below_threshold",
        "prediction_horizon_days": 30,
        "model_type": "LogisticRegression",
        "calibrated_probability": False,
        "data_type": "synthetic",
    }

    tools = MaintenanceTools()

    with patch(
        "aeromaintain.agents.tools.get_latest_component_features",
        return_value=fake_features,
    ) as feature_mock:
        with patch.object(
            tools,
            "_get_predictor",
        ) as predictor_mock:
            predictor_mock.return_value.predict_one.return_value = (
                fake_prediction
            )

            result = tools.calculate_failure_risk(
                "CMP-00196"
            )

    feature_mock.assert_called_once_with(
        "CMP-00196"
    )

    predictor_mock.return_value.predict_one.assert_called_once_with(
        fake_features
    )

    assert result == {
        "component_id": "CMP-00196",
        "risk_score": 0.71,
        "threshold": 0.72,
        "above_threshold": False,
        "interpretation": "below_threshold",
        "prediction_horizon_days": 30,
        "model_type": "LogisticRegression",
        "calibrated_probability": False,
        "data_type": "synthetic",
    }


def test_calculate_failure_risk_rejects_empty_component_id():
    """
    Empty component identifiers should be rejected before feature
    engineering or model inference begins.
    """
    tools = MaintenanceTools()

    with pytest.raises(
        ValueError,
        match="component_id must not be empty",
    ):
        tools.calculate_failure_risk(
            component_id="",
        )