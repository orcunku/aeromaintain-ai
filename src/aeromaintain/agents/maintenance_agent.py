from typing import Any

from aeromaintain.agents.tools import MaintenanceTools


class MaintenanceAgent:
    """
    Evidence-driven maintenance investigation agent.

    The agent coordinates controlled AeroMaintain tools and combines:
    - predictive-maintenance risk scoring
    - historical maintenance events
    - synthetic maintenance knowledge-base evidence
    - deterministic recommendation logic
    - human-gated action proposals

    The agent can recommend and propose workflow actions, but it does not
    execute real maintenance actions, create operational work orders,
    ground aircraft, release aircraft, or replace components.

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

        Investigation flow:
        1. observe predictive risk
        2. inspect maintenance history
        3. retrieve relevant engineering evidence
        4. recommend the next investigation step
        5. propose a human-gated workflow action
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

        recommendation = self._build_recommendation(
            risk=risk,
            history=history,
            evidence=evidence,
        )

        action_proposal = self._build_action_proposal(
            component_id=component_id,
            component_type=component_type,
            recommendation=recommendation,
        )

        summary = self._build_summary(
            component_id=component_id,
            component_type=component_type,
            risk=risk,
            history=history,
            evidence=evidence,
            recommendation=recommendation,
            action_proposal=action_proposal,
        )

        return {
            "component_id": component_id,
            "component_type": component_type,
            "risk": risk,
            "maintenance_history": history,
            "retrieval_query": retrieval_query,
            "evidence": evidence,
            "recommendation": recommendation,
            "action_proposal": action_proposal,
            "summary": summary,
            "limitations": {
                "data_type": "synthetic",
                "autonomous_maintenance_decision": False,
                "real_world_action_execution": False,
                "human_approval_required": True,
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
    def _build_recommendation(
        risk: dict,
        history: list[dict],
        evidence: list[dict],
    ) -> dict[str, Any]:
        """
        Build a deterministic next-step recommendation.

        This is workflow recommendation logic, not an autonomous
        aircraft-maintenance decision.
        """
        reasons: list[str] = []

        if risk["above_threshold"]:
            recommendation_code = "ENGINEERING_REVIEW"
            priority = "ELEVATED"

            reasons.append(
                "Predictive risk score is above the configured "
                "investigation threshold."
            )
        else:
            recommendation_code = "CONDITION_MONITORING"
            priority = "ROUTINE"

            reasons.append(
                "Predictive risk score is below the configured "
                "investigation threshold."
            )

        if history:
            reasons.append(
                "Maintenance history is available for contextual review."
            )
        else:
            reasons.append(
                "No maintenance events were returned for the "
                "configured history window."
            )

        if evidence:
            reasons.append(
                "Relevant synthetic engineering evidence was retrieved "
                "for investigation support."
            )
        else:
            reasons.append(
                "No supporting knowledge-base evidence was retrieved."
            )

        return {
            "recommendation_code": recommendation_code,
            "priority": priority,
            "recommended_next_step": (
                "Route the component investigation to a qualified "
                "human reviewer for engineering assessment."
                if risk["above_threshold"]
                else
                "Continue condition monitoring and retain the "
                "investigation record for human review."
            ),
            "reasons": reasons,
            "decision_basis": [
                "predictive_risk",
                "maintenance_history",
                "retrieved_evidence",
            ],
            "autonomous_decision": False,
        }

    @staticmethod
    def _build_action_proposal(
        component_id: str,
        component_type: str,
        recommendation: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convert the recommendation into a human-gated workflow proposal.

        No external system is modified and no real-world maintenance
        action is executed.
        """
        if (
            recommendation["recommendation_code"]
            == "ENGINEERING_REVIEW"
        ):
            action_type = "CREATE_ENGINEERING_REVIEW_CASE"
        else:
            action_type = "CREATE_MONITORING_CASE"

        return {
            "action_type": action_type,
            "component_id": component_id,
            "component_type": component_type,
            "priority": recommendation["priority"],
            "status": "PROPOSED",
            "approval_status": "HUMAN_APPROVAL_REQUIRED",
            "execution_status": "NOT_EXECUTED",
            "external_system_modified": False,
            "description": (
                "Proposed workflow action generated from the "
                "investigation. A qualified human must review and "
                "approve any operational follow-up."
            ),
        }

    @staticmethod
    def _build_summary(
        component_id: str,
        component_type: str,
        risk: dict,
        history: list[dict],
        evidence: list[dict],
        recommendation: dict[str, Any],
        action_proposal: dict[str, Any],
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
            "recommendation_code": recommendation[
                "recommendation_code"
            ],
            "recommendation_priority": recommendation[
                "priority"
            ],
            "proposed_action": action_proposal[
                "action_type"
            ],
            "action_status": action_proposal[
                "status"
            ],
            "approval_status": action_proposal[
                "approval_status"
            ],
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
        "Recommendation:",
        investigation["recommendation"][
            "recommendation_code"
        ],
    )

    print(
        "Priority:",
        investigation["recommendation"][
            "priority"
        ],
    )

    print(
        "Proposed action:",
        investigation["action_proposal"][
            "action_type"
        ],
    )

    print(
        "Approval status:",
        investigation["action_proposal"][
            "approval_status"
        ],
    )

    print(
        "Execution status:",
        investigation["action_proposal"][
            "execution_status"
        ],
    )