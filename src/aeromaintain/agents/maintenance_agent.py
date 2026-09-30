from typing import Any

from aeromaintain.agents.tools import MaintenanceTools


class MaintenanceAgent:
    """
    Evidence-driven maintenance investigation agent.

    The agent coordinates the controlled AeroMaintain tools and combines:
    - predictive-maintenance risk scoring
    - historical maintenance events
    - synthetic maintenance knowledge-base evidence

    The agent does not make autonomous maintenance decisions.

    All data and maintenance documents used by this project are synthetic.
    Predictive risk scores are not calibrated real-world aircraft failure
    probabilities.
    """

    def __init__(
        self,
        tools: MaintenanceTools | None = None,
    ) -> None:
        self.tools = (
            tools
            if tools is not None
            else MaintenanceTools()
        )

    def investigate_component(
        self,
        component_id: str,
        top_k_documents: int = 3,
        history_limit: int = 5,
    ) -> dict[str, Any]:
        """
        Run a structured investigation for one synthetic component.

        The investigation combines:
        1. predictive-maintenance risk
        2. maintenance history
        3. relevant maintenance knowledge-base evidence
        """
        if not component_id.strip():
            raise ValueError(
                "component_id must not be empty."
            )

        if top_k_documents < 1:
            raise ValueError(
                "top_k_documents must be at least 1."
            )

        if history_limit < 1:
            raise ValueError(
                "history_limit must be at least 1."
            )

        risk = self.tools.calculate_failure_risk(
            component_id
        )

        history = self.tools.get_component_history(
            component_id=component_id,
            limit=history_limit,
        )

        component_type = self._resolve_component_type(
            history=history,
            component_id=component_id,
        )

        retrieval_query = self._build_retrieval_query(
            component_type=component_type,
            risk=risk,
            history=history,
        )

        evidence = self.tools.search_maintenance_docs(
            query=retrieval_query,
            top_k=top_k_documents,
            component_type=component_type,
        )

        summary = self._build_summary(
            component_id=component_id,
            component_type=component_type,
            risk=risk,
            history=history,
            evidence=evidence,
        )

        return {
            "component_id": component_id,
            "component_type": component_type,
            "risk": risk,
            "maintenance_history": history,
            "retrieval_query": retrieval_query,
            "evidence": evidence,
            "summary": summary,
            "limitations": {
                "data_type": "synthetic",
                "autonomous_maintenance_decision": False,
                "calibrated_failure_probability": False,
                "approved_maintenance_guidance": False,
            },
        }

    def _resolve_component_type(
        self,
        history: list[dict],
        component_id: str,
    ) -> str:
        """
        Resolve the component type from maintenance history.

        If no maintenance event exists, use the latest local feature
        record through the predictive-risk tool's feature source.
        """
        if history:
            return str(
                history[0]["component_type"]
            )

        from aeromaintain.features.build_features import (
            get_latest_component_features,
        )

        features = get_latest_component_features(
            component_id
        )

        return str(
            features["component_type"]
        )

    @staticmethod
    def _build_retrieval_query(
        component_type: str,
        risk: dict,
        history: list[dict],
    ) -> str:
        """
        Construct a deterministic retrieval query from available evidence.
        """
        query_parts = [
            component_type.replace("_", " "),
            "condition indicators",
            "investigation workflow",
        ]

        if risk["above_threshold"]:
            query_parts.extend(
                [
                    "elevated predictive maintenance risk",
                    "abnormal condition",
                ]
            )

        if history:
            latest_event = history[0]

            fault_code = latest_event.get(
                "fault_code"
            )

            technician_note = latest_event.get(
                "technician_note"
            )

            corrective_action = latest_event.get(
                "corrective_action"
            )

            if fault_code:
                query_parts.append(
                    str(fault_code)
                )

            if technician_note:
                query_parts.append(
                    str(technician_note)
                )

            if corrective_action:
                query_parts.append(
                    str(corrective_action)
                )

        return " ".join(query_parts)

    @staticmethod
    def _build_summary(
        component_id: str,
        component_type: str,
        risk: dict,
        history: list[dict],
        evidence: list[dict],
    ) -> dict[str, Any]:
        """
        Build a factual structured summary from tool outputs.

        No LLM-generated maintenance instruction is produced here.
        """
        latest_history = (
            history[0]
            if history
            else None
        )

        evidence_sections = [
            item["section"]
            for item in evidence
        ]

        return {
            "component_id": component_id,
            "component_type": component_type,
            "risk_score": risk["risk_score"],
            "risk_threshold": risk["threshold"],
            "above_threshold": risk[
                "above_threshold"
            ],
            "risk_interpretation": risk[
                "interpretation"
            ],
            "prediction_horizon_days": risk[
                "prediction_horizon_days"
            ],
            "maintenance_event_count_returned": len(
                history
            ),
            "latest_maintenance_event": latest_history,
            "evidence_count": len(
                evidence
            ),
            "evidence_sections": evidence_sections,
        }


if __name__ == "__main__":
    agent = MaintenanceAgent()

    investigation = agent.investigate_component(
        component_id="CMP-00196",
        top_k_documents=3,
        history_limit=5,
    )

    print(
        "AeroMaintain investigation completed."
    )

    print(
        "Component:",
        investigation["component_id"],
    )

    print(
        "Component type:",
        investigation["component_type"],
    )

    print(
        "Risk score:",
        round(
            investigation["risk"]["risk_score"],
            4,
        ),
    )

    print(
        "Threshold:",
        round(
            investigation["risk"]["threshold"],
            4,
        ),
    )

    print(
        "Above threshold:",
        investigation["risk"][
            "above_threshold"
        ],
    )

    print(
        "Maintenance events:",
        len(
            investigation[
                "maintenance_history"
            ]
        ),
    )

    print(
        "Retrieved evidence:",
        len(
            investigation["evidence"]
        ),
    )

    print(
        "Evidence sections:",
        investigation["summary"][
            "evidence_sections"
        ],
    )

    print(
        "Synthetic decision-support only:",
        not investigation[
            "limitations"
        ][
            "autonomous_maintenance_decision"
        ],
    )