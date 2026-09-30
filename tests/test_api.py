from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from aeromaintain.api.main import app


def build_fake_investigation() -> dict:
    """
    Return a deterministic synthetic investigation response for API tests.
    """
    return {
        "component_id": "CMP-00196",
        "component_type": "HYDRAULIC_PUMP",
        "risk": {
            "component_id": "CMP-00196",
            "risk_score": 0.71,
            "threshold": 0.72,
            "above_threshold": False,
            "interpretation": "below_threshold",
            "prediction_horizon_days": 30,
            "model_type": "LogisticRegression",
            "calibrated_probability": False,
            "data_type": "synthetic",
        },
        "maintenance_history": [],
        "retrieval_query": (
            "HYDRAULIC PUMP condition indicators "
            "investigation workflow"
        ),
        "evidence": [],
        "summary": {
            "component_id": "CMP-00196",
            "component_type": "HYDRAULIC_PUMP",
            "risk_score": 0.71,
            "risk_threshold": 0.72,
            "above_threshold": False,
            "risk_interpretation": "below_threshold",
            "prediction_horizon_days": 30,
            "maintenance_event_count_returned": 0,
            "latest_maintenance_event": None,
            "evidence_count": 0,
            "evidence_sections": [],
        },
        "limitations": {
            "data_type": "synthetic",
            "autonomous_maintenance_decision": False,
            "calibrated_failure_probability": False,
            "approved_maintenance_guidance": False,
        },
    }


def test_root_endpoint():
    """
    Root endpoint should expose basic API metadata.
    """
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200

    assert response.json() == {
        "name": "AeroMaintain AI",
        "status": "running",
        "data_type": "synthetic",
    }


def test_health_endpoint():
    """
    Health endpoint should provide a lightweight readiness response.
    """
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
    }


def test_investigate_endpoint_returns_agent_result():
    """
    Investigation endpoint should pass request parameters to the
    maintenance agent and return its structured result.
    """
    fake_result = build_fake_investigation()

    with patch(
        "aeromaintain.api.main.MaintenanceAgent"
    ) as agent_class:
        fake_agent = Mock()
        fake_agent.investigate_component.return_value = (
            fake_result
        )
        agent_class.return_value = fake_agent

        with TestClient(app) as client:
            response = client.get(
                "/investigate/CMP-00196",
                params={
                    "top_k_documents": 2,
                    "history_limit": 4,
                },
            )

    assert response.status_code == 200
    assert response.json() == fake_result

    fake_agent.investigate_component.assert_called_once_with(
        component_id="CMP-00196",
        top_k_documents=2,
        history_limit=4,
    )


def test_investigate_endpoint_converts_value_error_to_400():
    """
    Invalid investigation input should be exposed as HTTP 400.
    """
    with patch(
        "aeromaintain.api.main.MaintenanceAgent"
    ) as agent_class:
        fake_agent = Mock()

        fake_agent.investigate_component.side_effect = (
            ValueError(
                "Unknown component_id: CMP-UNKNOWN"
            )
        )

        agent_class.return_value = fake_agent

        with TestClient(app) as client:
            response = client.get(
                "/investigate/CMP-UNKNOWN"
            )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "Unknown component_id: CMP-UNKNOWN"
        )
    }


def test_investigate_endpoint_reports_missing_model():
    """
    Missing local model artifacts should become HTTP 503 rather than
    an unhandled server error.
    """
    with patch(
        "aeromaintain.api.main.MaintenanceAgent"
    ) as agent_class:
        fake_agent = Mock()

        fake_agent.investigate_component.side_effect = (
            FileNotFoundError(
                "predictive_maintenance_bundle.joblib"
            )
        )

        agent_class.return_value = fake_agent

        with TestClient(app) as client:
            response = client.get(
                "/investigate/CMP-00196"
            )

    assert response.status_code == 503

    assert (
        "Predictive model artifact is unavailable"
        in response.json()["detail"]
    )


def test_risk_endpoint_returns_predictive_result():
    """
    Risk endpoint should expose the predictive-maintenance tool result.
    """
    fake_risk = {
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

    with patch(
        "aeromaintain.api.main.MaintenanceAgent"
    ) as agent_class:
        fake_agent = Mock()
        fake_agent.tools.calculate_failure_risk.return_value = (
            fake_risk
        )
        agent_class.return_value = fake_agent

        with TestClient(app) as client:
            response = client.get(
                "/components/CMP-00196/risk"
            )

    assert response.status_code == 200
    assert response.json() == fake_risk

    fake_agent.tools.calculate_failure_risk.assert_called_once_with(
        "CMP-00196"
    )


def test_risk_endpoint_converts_value_error_to_400():
    """
    Invalid component identifiers should become HTTP 400.
    """
    with patch(
        "aeromaintain.api.main.MaintenanceAgent"
    ) as agent_class:
        fake_agent = Mock()

        fake_agent.tools.calculate_failure_risk.side_effect = (
            ValueError(
                "Unknown component_id: CMP-UNKNOWN"
            )
        )

        agent_class.return_value = fake_agent

        with TestClient(app) as client:
            response = client.get(
                "/components/CMP-UNKNOWN/risk"
            )

    assert response.status_code == 400

    assert response.json() == {
        "detail": (
            "Unknown component_id: CMP-UNKNOWN"
        )
    }