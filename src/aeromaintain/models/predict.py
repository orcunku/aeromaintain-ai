from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from aeromaintain.config import PROJECT_ROOT


DEFAULT_MODEL_PATH = (
    PROJECT_ROOT
    / "artifacts"
    / "predictive_maintenance_bundle.joblib"
)


class PredictiveMaintenanceModel:
    """
    Local inference wrapper for the AeroMaintain predictive-maintenance model.

    The underlying classifier was trained on synthetic aircraft-maintenance
    data. Its predict_proba output is treated as a risk score, not as a
    calibrated real-world failure probability.

    The model estimates relative risk for a synthetic 30-day predictive
    maintenance horizon.
    """

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL_PATH,
    ) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                "Predictive-maintenance model artifact was not found at: "
                f"{self.model_path}"
            )

        bundle = joblib.load(self.model_path)

        self._validate_bundle(bundle)

        self.model = bundle["model"]
        self.threshold = float(bundle["threshold"])
        self.model_type = str(bundle["model_type"])
        self.prediction_horizon_days = int(
            bundle["prediction_horizon_days"]
        )
        self.feature_columns = list(
            bundle["feature_columns"]
        )

    @staticmethod
    def _validate_bundle(
        bundle: Any,
    ) -> None:
        """
        Validate the structure of the exported model bundle.
        """
        if not isinstance(bundle, dict):
            raise ValueError(
                "Model artifact must contain a dictionary bundle."
            )

        required_keys = {
            "model",
            "threshold",
            "model_type",
            "prediction_horizon_days",
            "feature_columns",
        }

        missing_keys = required_keys - set(bundle)

        if missing_keys:
            missing = ", ".join(
                sorted(missing_keys)
            )

            raise ValueError(
                "Model bundle is missing required keys: "
                f"{missing}"
            )

        model = bundle["model"]

        if not hasattr(model, "predict_proba"):
            raise ValueError(
                "Loaded model does not provide predict_proba()."
            )

        feature_columns = bundle["feature_columns"]

        if not isinstance(
            feature_columns,
            (list, tuple),
        ):
            raise ValueError(
                "feature_columns must be a list or tuple."
            )

        if not feature_columns:
            raise ValueError(
                "feature_columns must not be empty."
            )

    def _prepare_features(
        self,
        features: dict[str, Any] | pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Convert input features into the exact schema expected by the model.
        """
        if isinstance(features, dict):
            frame = pd.DataFrame(
                [features]
            )

        elif isinstance(features, pd.DataFrame):
            frame = features.copy()

        else:
            raise TypeError(
                "features must be a dictionary or pandas DataFrame."
            )

        if frame.empty:
            raise ValueError(
                "Feature input must contain at least one row."
            )

        missing_features = [
            column
            for column in self.feature_columns
            if column not in frame.columns
        ]

        if missing_features:
            missing = ", ".join(
                missing_features
            )

            raise ValueError(
                "Missing required model features: "
                f"{missing}"
            )

        return frame[
            self.feature_columns
        ].copy()

    def predict(
        self,
        features: dict[str, Any] | pd.DataFrame,
    ) -> list[dict[str, Any]]:
        """
        Calculate predictive-maintenance risk scores.

        Important:
        risk_score is the classifier's raw predict_proba score.
        Because the model has not been probability-calibrated, this value
        should not be interpreted as a literal probability of failure.
        """
        feature_frame = self._prepare_features(
            features
        )

        risk_scores = self.model.predict_proba(
            feature_frame
        )[:, 1]

        predictions = []

        for risk_score in risk_scores:
            score = float(risk_score)

            predictions.append(
                {
                    "risk_score": score,
                    "threshold": self.threshold,
                    "above_threshold": (
                        score >= self.threshold
                    ),
                    "prediction_horizon_days": (
                        self.prediction_horizon_days
                    ),
                    "model_type": self.model_type,
                    "interpretation": (
                        "elevated"
                        if score >= self.threshold
                        else "below_threshold"
                    ),
                    "calibrated_probability": False,
                    "data_type": "synthetic",
                }
            )

        return predictions

    def predict_one(
        self,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Convenience method for scoring one feature record.
        """
        predictions = self.predict(
            features
        )

        return predictions[0]

    def model_info(
        self,
    ) -> dict[str, Any]:
        """
        Return metadata about the locally loaded model.
        """
        return {
            "model_type": self.model_type,
            "threshold": self.threshold,
            "prediction_horizon_days": (
                self.prediction_horizon_days
            ),
            "feature_count": len(
                self.feature_columns
            ),
            "feature_columns": (
                self.feature_columns.copy()
            ),
            "model_path": str(
                self.model_path
            ),
            "calibrated_probability": False,
            "data_type": "synthetic",
        }


if __name__ == "__main__":
    predictor = PredictiveMaintenanceModel()

    info = predictor.model_info()

    print(
        "Predictive-maintenance model loaded successfully."
    )

    print(
        "Model type:",
        info["model_type"],
    )

    print(
        "Threshold:",
        info["threshold"],
    )

    print(
        "Prediction horizon:",
        info["prediction_horizon_days"],
        "days",
    )

    print(
        "Feature count:",
        info["feature_count"],
    )