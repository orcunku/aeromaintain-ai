from dataclasses import asdict

import pandas as pd

from aeromaintain.config import RAW_DATA_DIR
from aeromaintain.features.build_features import (
    get_latest_component_features,
)
from aeromaintain.models.predict import (
    PredictiveMaintenanceModel,
)
from aeromaintain.rag.retrieve import (
    LSAMaintenanceRetriever,
)


class MaintenanceTools:
    """
    Controlled tool layer used by the AeroMaintain maintenance agent.

    The tools provide structured access to:
    - synthetic maintenance knowledge-base retrieval
    - historical component maintenance events
    - predictive-maintenance risk scoring

    All project data is synthetic. Predictive model scores are intended
    for decision-support demonstrations only and are not calibrated
    real-world aircraft failure probabilities.
    """

    def __init__(self) -> None:
        self.retriever = LSAMaintenanceRetriever(
            n_components=10,
        )

        self.maintenance_df = pd.read_parquet(
            RAW_DATA_DIR / "maintenance.parquet"
        )

        self.maintenance_df["timestamp"] = pd.to_datetime(
            self.maintenance_df["timestamp"]
        )

        # Load lazily when predictive scoring is first requested.
        # This allows retrieval/history tools to remain usable even
        # when a local model artifact has not been installed.
        self._predictor: PredictiveMaintenanceModel | None = None

    def _get_predictor(
        self,
    ) -> PredictiveMaintenanceModel:
        """
        Load and cache the local predictive-maintenance model.
        """
        if self._predictor is None:
            self._predictor = PredictiveMaintenanceModel()

        return self._predictor

    def search_maintenance_docs(
        self,
        query: str,
        top_k: int = 3,
        component_type: str | None = None,
    ) -> list[dict]:
        """
        Search the synthetic maintenance knowledge base.

        Returns structured evidence that can later be consumed by an
        agent or API without exposing internal retriever objects.
        """
        results = self.retriever.retrieve(
            query=query,
            top_k=top_k,
            component_type=component_type,
        )

        evidence = []

        for result in results:
            chunk_data = asdict(result.chunk)

            evidence.append(
                {
                    "score": float(result.score),
                    "document_id": chunk_data["document_id"],
                    "title": chunk_data["title"],
                    "component_type": chunk_data["component_type"],
                    "section": chunk_data["section"],
                    "content": chunk_data["content"],
                    "source_file": chunk_data["source_file"],
                }
            )

        return evidence

    def get_component_history(
        self,
        component_id: str,
        limit: int = 10,
    ) -> list[dict]:
        """
        Return historical maintenance events for one component.

        Events are returned newest first.
        """
        if not component_id.strip():
            raise ValueError(
                "component_id must not be empty."
            )

        if limit < 1:
            raise ValueError(
                "limit must be at least 1."
            )

        component_history = self.maintenance_df[
            self.maintenance_df["component_id"] == component_id
        ].copy()

        component_history = component_history.sort_values(
            "timestamp",
            ascending=False,
        ).head(limit)

        history = []

        for row in component_history.itertuples(
            index=False
        ):
            history.append(
                {
                    "work_order_id": row.work_order_id,
                    "timestamp": row.timestamp.isoformat(),
                    "aircraft_id": row.aircraft_id,
                    "component_id": row.component_id,
                    "component_type": row.component_type,
                    "event_type": row.event_type,
                    "fault_code": row.fault_code,
                    "technician_note": row.technician_note,
                    "corrective_action": row.corrective_action,
                }
            )

        return history

    def calculate_failure_risk(
        self,
        component_id: str,
    ) -> dict:
        """
        Calculate the latest predictive-maintenance risk score for one
        synthetic component.

        The returned risk_score is the classifier's raw predict_proba
        score. It is not a calibrated real-world failure probability.
        """
        if not component_id.strip():
            raise ValueError(
                "component_id must not be empty."
            )

        features = get_latest_component_features(
            component_id
        )

        predictor = self._get_predictor()

        prediction = predictor.predict_one(
            features
        )

        return {
            "component_id": component_id,
            "risk_score": prediction["risk_score"],
            "threshold": prediction["threshold"],
            "above_threshold": prediction["above_threshold"],
            "interpretation": prediction["interpretation"],
            "prediction_horizon_days": prediction[
                "prediction_horizon_days"
            ],
            "model_type": prediction["model_type"],
            "calibrated_probability": prediction[
                "calibrated_probability"
            ],
            "data_type": prediction["data_type"],
        }