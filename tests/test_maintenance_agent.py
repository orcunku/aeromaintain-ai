from unittest.mock import Mock

import pytest

from aeromaintain.agents.maintenance_agent import (
    MaintenanceAgent,
)


def build_mock_tools() -> Mock:
    """
    Create deterministic tool outputs for agent unit tests.
    """
    tools = Mock()

    tools.calculate_failure_risk.return_value = {
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

    tools.get_component_history.return_value = [
        {
            "work_order_id": "WO-000002",
            "timestamp": "2025-01-06T00:00:00",
            "aircraft_id": "AC-040",
            "component_id": "CMP-00196",
            "component_type": "HYDRAULIC_PUMP",
            "event_type": "CORRECTIVE",
            "fault_code": "FLT-761",
            "technician_note": (
                "Inspection performed following abnormal "
                "HYDRAULIC_PUMP condition."
            ),
            "corrective_action": "REPAIRED",
        }
    ]

    tools.search_maintenance_docs.return_value = [
        {
            "score": 0.95,
            "document_id": "AM-KB-HYD-001",
            "title": "Hydraulic Pump",
            "component_type": "HYDRAULIC_PUMP",
            "section": "Typical Condition Indicators",
            "content": "Synthetic condition evidence.",
            "source_file": "am-kb-hyd-001.md",
        },
        {
            "score": 0.85,
            "document_id": "AM-KB-HYD-001",
            "title": "Hydraulic Pump",
            "component_type": "HYDRAULIC_PUMP",
            "section": "Investigation Workflow",
            "content": "Synthetic investigation evidence.",
            "source_file": "am-kb-hyd-001.md",
        },
    ]

    return tools


def test_investigate_component_combines_tool_outputs():
    """
    The agent should combine predictive risk, maintenance history,
    retrieved evidence, recommendation, and action proposal.
    """
    tools = build_mock_tools()

    agent = MaintenanceAgent(
        tools=tools,
    )

    result = agent.investigate_component(
        component_id="CMP-00196",
        top_k_documents=2,
        history_limit=5,
    )

    assert result["component_id"] == "CMP-00196"
    assert result["component_type"] == "HYDRAULIC_PUMP"

    assert result["risk"]["risk_score"] == 0.71

    assert len(
        result["maintenance_history"]
    ) == 1

    assert len(
        result["evidence"]
    ) == 2

    assert (
        result["summary"]["risk_score"]
        == 0.71
    )

    assert (
        result["summary"][
            "maintenance_event_count_returned"
        ]
        == 1
    )

    assert (
        result["summary"]["evidence_count"]
        == 2
    )

    assert "recommendation" in result
    assert "action_proposal" in result


def test_investigate_component_calls_expected_tools():
    """
    The agent should call each controlled tool with the expected
    component and investigation limits.
    """
    tools = build_mock_tools()

    agent = MaintenanceAgent(
        tools=tools,
    )

    agent.investigate_component(
        component_id="CMP-00196",
        top_k_documents=2,
        history_limit=5,
    )

    tools.calculate_failure_risk.assert_called_once_with(
        "CMP-00196"
    )

    tools.get_component_history.assert_called_once_with(
        component_id="CMP-00196",
        limit=5,
    )

    tools.search_maintenance_docs.assert_called_once()

    call_kwargs = (
        tools.search_maintenance_docs.call_args.kwargs
    )

    assert call_kwargs["top_k"] == 2

    assert (
        call_kwargs["component_type"]
        == "HYDRAULIC_PUMP"
    )


def test_retrieval_query_uses_maintenance_context():
    """
    The retrieval query should include component and historical
    maintenance context rather than using a fixed generic query.
    """
    tools = build_mock_tools()

    agent = MaintenanceAgent(
        tools=tools,
    )

    result = agent.investigate_component(
        component_id="CMP-00196",
    )

    query = result[
        "retrieval_query"
    ].lower()

    assert "hydraulic pump" in query
    assert "flt-761" in query
    assert "repaired" in query
    assert "investigation workflow" in query


def test_elevated_risk_adds_risk_context_to_query():
    """
    Above-threshold predictions should add elevated-risk context
    to the knowledge-base retrieval query.
    """
    tools = build_mock_tools()

    tools.calculate_failure_risk.return_value[
        "above_threshold"
    ] = True

    tools.calculate_failure_risk.return_value[
        "interpretation"
    ] = "elevated"

    agent = MaintenanceAgent(
        tools=tools,
    )

    result = agent.investigate_component(
        component_id="CMP-00196",
    )

    query = result[
        "retrieval_query"
    ].lower()

    assert (
        "elevated predictive maintenance risk"
        in query
    )

    assert "abnormal condition" in query


def test_below_threshold_proposes_monitoring_case():
    """
    Below-threshold risk should produce a routine monitoring
    recommendation and a non-executed monitoring-case proposal.
    """
    tools = build_mock_tools()

    agent = MaintenanceAgent(
        tools=tools,
    )

    result = agent.investigate_component(
        component_id="CMP-00196",
    )

    recommendation = result[
        "recommendation"
    ]

    proposal = result[
        "action_proposal"
    ]

    assert (
        recommendation["recommendation_code"]
        == "CONDITION_MONITORING"
    )

    assert (
        recommendation["priority"]
        == "ROUTINE"
    )

    assert (
        recommendation["autonomous_decision"]
        is False
    )

    assert (
        proposal["action_type"]
        == "CREATE_MONITORING_CASE"
    )

    assert proposal["status"] == "PROPOSED"

    assert (
        proposal["approval_status"]
        == "HUMAN_APPROVAL_REQUIRED"
    )

    assert (
        proposal["execution_status"]
        == "NOT_EXECUTED"
    )

    assert (
        proposal["external_system_modified"]
        is False
    )


def test_above_threshold_proposes_engineering_review():
    """
    Above-threshold risk should escalate the recommendation to
    a human engineering-review workflow proposal.
    """
    tools = build_mock_tools()

    tools.calculate_failure_risk.return_value.update(
        {
            "risk_score": 0.84,
            "above_threshold": True,
            "interpretation": "elevated",
        }
    )

    agent = MaintenanceAgent(
        tools=tools,
    )

    result = agent.investigate_component(
        component_id="CMP-00196",
    )

    recommendation = result[
        "recommendation"
    ]

    proposal = result[
        "action_proposal"
    ]

    assert (
        recommendation["recommendation_code"]
        == "ENGINEERING_REVIEW"
    )

    assert (
        recommendation["priority"]
        == "ELEVATED"
    )

    assert (
        proposal["action_type"]
        == "CREATE_ENGINEERING_REVIEW_CASE"
    )

    assert (
        proposal["priority"]
        == "ELEVATED"
    )

    assert (
        proposal["approval_status"]
        == "HUMAN_APPROVAL_REQUIRED"
    )

    assert (
        proposal["execution_status"]
        == "NOT_EXECUTED"
    )

    assert (
        proposal["external_system_modified"]
        is False
    )


def test_summary_exposes_agentic_workflow_state():
    """
    The structured summary should expose the recommendation and
    human-gated action state for downstream API/UI consumers.
    """
    tools = build_mock_tools()

    agent = MaintenanceAgent(
        tools=tools,
    )

    result = agent.investigate_component(
        component_id="CMP-00196",
    )

    summary = result["summary"]

    assert (
        summary["recommendation_code"]
        == "CONDITION_MONITORING"
    )

    assert (
        summary["recommendation_priority"]
        == "ROUTINE"
    )

    assert (
        summary["proposed_action"]
        == "CREATE_MONITORING_CASE"
    )

    assert (
        summary["action_status"]
        == "PROPOSED"
    )

    assert (
        summary["approval_status"]
        == "HUMAN_APPROVAL_REQUIRED"
    )


def test_investigation_exposes_safety_limitations():
    """
    The agent output should explicitly state the synthetic,
    human-gated, and decision-support limitations.
    """
    tools = build_mock_tools()

    agent = MaintenanceAgent(
        tools=tools,
    )

    result = agent.investigate_component(
        component_id="CMP-00196",
    )

    limitations = result[
        "limitations"
    ]

    assert limitations["data_type"] == "synthetic"

    assert (
        limitations[
            "autonomous_maintenance_decision"
        ]
        is False
    )

    assert (
        limitations[
            "real_world_action_execution"
        ]
        is False
    )

    assert (
        limitations[
            "human_approval_required"
        ]
        is True
    )

    assert (
        limitations[
            "calibrated_failure_probability"
        ]
        is False
    )

    assert (
        limitations[
            "approved_maintenance_guidance"
        ]
        is False
    )


def test_investigate_component_rejects_empty_component_id():
    """
    Empty component identifiers should be rejected before tools
    are called.
    """
    tools = build_mock_tools()

    agent = MaintenanceAgent(
        tools=tools,
    )

    with pytest.raises(
        ValueError,
        match="component_id must not be empty",
    ):
        agent.investigate_component(
            component_id="",
        )

    tools.calculate_failure_risk.assert_not_called()


def test_investigate_component_rejects_invalid_document_limit():
    """
    Document retrieval must request at least one result.
    """
    tools = build_mock_tools()

    agent = MaintenanceAgent(
        tools=tools,
    )

    with pytest.raises(
        ValueError,
        match="top_k_documents must be at least 1",
    ):
        agent.investigate_component(
            component_id="CMP-00196",
            top_k_documents=0,
        )


def test_investigate_component_rejects_invalid_history_limit():
    """
    Maintenance history must request at least one event.
    """
    tools = build_mock_tools()

    agent = MaintenanceAgent(
        tools=tools,
    )

    with pytest.raises(
        ValueError,
        match="history_limit must be at least 1",
    ):
        agent.investigate_component(
            component_id="CMP-00196",
            history_limit=0,
        )