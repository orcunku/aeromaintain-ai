"""
AeroMaintain AI - Predictive Maintenance ML Training

Trains and evaluates predictive-maintenance classification models
using the Gold feature table.

ML responsibilities:
- Use time-based train/validation/test splits.
- Apply purge gaps matching the 30-day prediction horizon.
- Fit preprocessing only on training data.
- Compare Logistic Regression and Random Forest baselines.
- Select the model using validation PR-AUC.
- Select the classification threshold using validation F1.
- Evaluate the selected model once on the holdout test set.
- Track parameters, metrics, tags, and the model with MLflow.

Synthetic data only. The resulting model is intended for
portfolio demonstration and decision-support experimentation,
not real aircraft maintenance decisions.
"""

import numpy as np
import pandas as pd

import mlflow
import mlflow.sklearn

from pyspark.sql import SparkSession
from pyspark.sql.functions import col

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)


CATALOG = "workspace"
SCHEMA = "aeromaintain"

GOLD_TABLE = (
    f"{CATALOG}.{SCHEMA}."
    "gold_predictive_features"
)

EXPERIMENT_NAME = (
    "/Shared/"
    "aeromaintain-ai-predictive-maintenance"
)

TARGET_COLUMN = "failure_within_30_days"

PREDICTION_HORIZON_DAYS = 30

TRAIN_END = "2025-08-01"

VALIDATION_START = "2025-09-01"
VALIDATION_END = "2025-09-30"

TEST_START = "2025-10-31"
TEST_END = "2025-12-01"


CATEGORICAL_FEATURES = [
    "aircraft_type",
    "component_type",
]


NUMERIC_FEATURES = [
    "age_years",
    "utilization_factor",
    "flight_cycles",
    "expected_life_cycles",
    "cycle_life_ratio",
    "days_since_maintenance",
    "previous_failures",
    "temperature_c",
    "vibration_mm_s",
    "temperature_missing_flag",
    "temperature_mean_7d",
    "temperature_mean_30d",
    "temperature_std_7d",
    "temperature_delta_7d",
    "vibration_mean_7d",
    "vibration_mean_30d",
    "vibration_std_7d",
    "vibration_delta_7d",
]


MODEL_FEATURE_COLUMNS = (
    CATEGORICAL_FEATURES
    + NUMERIC_FEATURES
)


def build_time_splits(
    gold_df,
):
    """
    Create purged temporal train, validation, and test splits.

    The omitted periods between splits provide 30-day gaps matching
    the prediction horizon. This reduces overlap between future
    label windows across model-development periods.
    """

    train_df = gold_df.filter(
        col("timestamp") <= TRAIN_END
    )

    validation_df = gold_df.filter(
        (col("timestamp") >= VALIDATION_START)
        & (col("timestamp") <= VALIDATION_END)
    )

    test_df = gold_df.filter(
        (col("timestamp") >= TEST_START)
        & (col("timestamp") <= TEST_END)
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


def to_pandas_dataset(
    dataframe,
):
    """
    Convert the selected model features and target to Pandas.

    This project dataset is intentionally small enough for
    scikit-learn model development in memory.
    """

    return (
        dataframe
        .select(
            *MODEL_FEATURE_COLUMNS,
            TARGET_COLUMN,
        )
        .toPandas()
    )


def build_preprocessor() -> ColumnTransformer:
    """
    Build preprocessing transformations.

    The returned transformer must be fit only on training data.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                ),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ]
    )


def build_logistic_regression() -> Pipeline:
    """
    Build the Logistic Regression baseline pipeline.
    """

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                ),
            ),
        ]
    )


def build_random_forest() -> Pipeline:
    """
    Build the Random Forest comparison pipeline.
    """

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_leaf=5,
                    class_weight=(
                        "balanced_subsample"
                    ),
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )


def find_best_f1_threshold(
    y_true,
    probabilities,
):
    """
    Select the classification threshold that maximizes F1
    on the validation set.
    """

    precision_values, recall_values, thresholds = (
        precision_recall_curve(
            y_true,
            probabilities,
        )
    )

    f1_values = (
        2
        * precision_values[:-1]
        * recall_values[:-1]
        / (
            precision_values[:-1]
            + recall_values[:-1]
            + 1e-12
        )
    )

    best_index = int(
        np.argmax(f1_values)
    )

    return {
        "threshold": float(
            thresholds[best_index]
        ),
        "precision": float(
            precision_values[best_index]
        ),
        "recall": float(
            recall_values[best_index]
        ),
        "f1": float(
            f1_values[best_index]
        ),
    }


def evaluate_probabilities(
    y_true,
    probabilities,
    threshold,
):
    """
    Calculate classification metrics for a fixed threshold.
    """

    predictions = (
        probabilities >= threshold
    ).astype(int)

    return {
        "pr_auc": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "f1": float(
            f1_score(
                y_true,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": (
            confusion_matrix(
                y_true,
                predictions,
            )
        ),
    }


def print_split_summary(
    name,
    y_values,
) -> None:
    """
    Print class balance for one temporal split.
    """

    row_count = len(y_values)
    positives = int(
        y_values.sum()
    )

    positive_rate = (
        positives
        / row_count
        * 100
    )

    print(
        f"{name}: "
        f"rows={row_count:,}, "
        f"positives={positives:,}, "
        f"positive_rate={positive_rate:.2f}%"
    )


def log_selected_model(
    model,
    threshold,
    validation_metrics,
    test_metrics,
    test_random_baseline,
) -> str:
    """
    Track the selected model and its evaluation in MLflow.
    """

    mlflow.set_experiment(
        EXPERIMENT_NAME
    )

    with mlflow.start_run(
        run_name="logistic-regression-baseline"
    ) as run:

        mlflow.log_params(
            {
                "model_type": (
                    "LogisticRegression"
                ),
                "class_weight": "balanced",
                "max_iter": 1000,
                "random_state": 42,
                "prediction_horizon_days": (
                    PREDICTION_HORIZON_DAYS
                ),
                "train_end": TRAIN_END,
                "validation_start": (
                    VALIDATION_START
                ),
                "validation_end": (
                    VALIDATION_END
                ),
                "test_start": TEST_START,
                "test_end": TEST_END,
                "numeric_feature_count": len(
                    NUMERIC_FEATURES
                ),
                "categorical_feature_count": len(
                    CATEGORICAL_FEATURES
                ),
            }
        )

        mlflow.log_metrics(
            {
                "validation_pr_auc": (
                    validation_metrics[
                        "pr_auc"
                    ]
                ),
                "validation_roc_auc": (
                    validation_metrics[
                        "roc_auc"
                    ]
                ),
                "selected_threshold": (
                    threshold
                ),
                "test_pr_auc": (
                    test_metrics[
                        "pr_auc"
                    ]
                ),
                "test_roc_auc": (
                    test_metrics[
                        "roc_auc"
                    ]
                ),
                "test_precision": (
                    test_metrics[
                        "precision"
                    ]
                ),
                "test_recall": (
                    test_metrics[
                        "recall"
                    ]
                ),
                "test_f1": (
                    test_metrics[
                        "f1"
                    ]
                ),
                "test_pr_auc_lift": (
                    test_metrics[
                        "pr_auc"
                    ]
                    / test_random_baseline
                ),
            }
        )

        mlflow.set_tags(
            {
                "project": "AeroMaintain AI",
                "data_type": "synthetic",
                "task": (
                    "30-day predictive "
                    "maintenance risk"
                ),
                "model_status": "baseline",
                "threshold_source": (
                    "validation_f1"
                ),
                "intended_use": (
                    "decision_support"
                ),
            }
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            name=(
                "predictive_maintenance_model"
            ),
        )

        run_id = run.info.run_id

    return run_id


def main() -> None:
    spark = (
        SparkSession
        .getActiveSession()
    )

    if spark is None:
        raise RuntimeError(
            "No active Spark session found. "
            "Run this script inside Databricks."
        )

    gold_df = spark.table(
        GOLD_TABLE
    )

    (
        train_df,
        validation_df,
        test_df,
    ) = build_time_splits(
        gold_df
    )

    train_pd = to_pandas_dataset(
        train_df
    )

    validation_pd = to_pandas_dataset(
        validation_df
    )

    test_pd = to_pandas_dataset(
        test_df
    )

    X_train = train_pd[
        MODEL_FEATURE_COLUMNS
    ]
    y_train = train_pd[
        TARGET_COLUMN
    ]

    X_validation = validation_pd[
        MODEL_FEATURE_COLUMNS
    ]
    y_validation = validation_pd[
        TARGET_COLUMN
    ]

    X_test = test_pd[
        MODEL_FEATURE_COLUMNS
    ]
    y_test = test_pd[
        TARGET_COLUMN
    ]

    print_split_summary(
        "Train",
        y_train,
    )

    print_split_summary(
        "Validation",
        y_validation,
    )

    print_split_summary(
        "Test",
        y_test,
    )

    logistic_model = (
        build_logistic_regression()
    )

    random_forest_model = (
        build_random_forest()
    )

    print(
        "\nTraining Logistic Regression..."
    )

    logistic_model.fit(
        X_train,
        y_train,
    )

    print(
        "Training Random Forest..."
    )

    random_forest_model.fit(
        X_train,
        y_train,
    )

    logistic_validation_probabilities = (
        logistic_model.predict_proba(
            X_validation
        )[:, 1]
    )

    random_forest_validation_probabilities = (
        random_forest_model.predict_proba(
            X_validation
        )[:, 1]
    )

    logistic_validation_pr_auc = (
        average_precision_score(
            y_validation,
            logistic_validation_probabilities,
        )
    )

    random_forest_validation_pr_auc = (
        average_precision_score(
            y_validation,
            random_forest_validation_probabilities,
        )
    )

    print(
        "\nValidation PR-AUC"
    )

    print(
        "Logistic Regression:",
        f"{logistic_validation_pr_auc:.4f}",
    )

    print(
        "Random Forest:",
        f"{random_forest_validation_pr_auc:.4f}",
    )

    if (
        logistic_validation_pr_auc
        < random_forest_validation_pr_auc
    ):
        raise RuntimeError(
            "Random Forest outperformed Logistic "
            "Regression. Review model-selection "
            "and MLflow logging logic before "
            "continuing."
        )

    selected_model = logistic_model

    threshold_metrics = (
        find_best_f1_threshold(
            y_validation,
            logistic_validation_probabilities,
        )
    )

    selected_threshold = (
        threshold_metrics[
            "threshold"
        ]
    )

    validation_metrics = (
        evaluate_probabilities(
            y_validation,
            logistic_validation_probabilities,
            selected_threshold,
        )
    )

    test_probabilities = (
        selected_model.predict_proba(
            X_test
        )[:, 1]
    )

    test_metrics = (
        evaluate_probabilities(
            y_test,
            test_probabilities,
            selected_threshold,
        )
    )

    test_random_baseline = float(
        y_test.mean()
    )

    print(
        "\nSelected threshold:",
        f"{selected_threshold:.4f}",
    )

    print(
        "Validation PR-AUC:",
        f"{validation_metrics['pr_auc']:.4f}",
    )

    print(
        "Validation ROC-AUC:",
        f"{validation_metrics['roc_auc']:.4f}",
    )

    print(
        "\nFinal Test Metrics"
    )

    print(
        "PR-AUC:",
        f"{test_metrics['pr_auc']:.4f}",
    )

    print(
        "ROC-AUC:",
        f"{test_metrics['roc_auc']:.4f}",
    )

    print(
        "Precision:",
        f"{test_metrics['precision']:.4f}",
    )

    print(
        "Recall:",
        f"{test_metrics['recall']:.4f}",
    )

    print(
        "F1:",
        f"{test_metrics['f1']:.4f}",
    )

    print(
        "PR-AUC lift:",
        (
            f"{test_metrics['pr_auc'] / test_random_baseline:.2f}x"
        ),
    )

    print(
        "Confusion matrix:"
    )

    print(
        test_metrics[
            "confusion_matrix"
        ]
    )

    run_id = log_selected_model(
        model=selected_model,
        threshold=selected_threshold,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        test_random_baseline=(
            test_random_baseline
        ),
    )

    print(
        "\nMLflow run ID:",
        run_id,
    )


if __name__ == "__main__":
    main()