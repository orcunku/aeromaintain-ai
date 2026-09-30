from typing import Final

import numpy as np
import pandas as pd

from aeromaintain.config import RAW_DATA_DIR


MODEL_FEATURE_COLUMNS: Final[list[str]] = [
    "aircraft_type",
    "component_type",
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


def load_raw_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Load the synthetic telemetry, component, and aircraft datasets.
    """
    telemetry_df = pd.read_parquet(
        RAW_DATA_DIR / "telemetry.parquet"
    )

    components_df = pd.read_parquet(
        RAW_DATA_DIR / "components.parquet"
    )

    aircraft_df = pd.read_parquet(
        RAW_DATA_DIR / "aircraft.parquet"
    )

    telemetry_df["timestamp"] = pd.to_datetime(
        telemetry_df["timestamp"]
    )

    return (
        telemetry_df,
        components_df,
        aircraft_df,
    )


def prepare_telemetry(
    telemetry_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare raw telemetry for local feature engineering.

    This mirrors the relevant Silver-layer behavior needed for
    inference:
    - remove duplicate component/timestamp observations
    - record whether temperature was originally missing

    Missing temperature values remain NaN because the trained sklearn
    pipeline contains its own numeric imputer.
    """
    dataframe = telemetry_df.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"]
    )

    dataframe = (
        dataframe
        .sort_values(
            [
                "component_id",
                "timestamp",
            ]
        )
        .drop_duplicates(
            subset=[
                "component_id",
                "timestamp",
            ],
            keep="first",
        )
        .reset_index(drop=True)
    )

    dataframe["temperature_missing_flag"] = (
        dataframe["temperature_c"]
        .isna()
        .astype(int)
    )

    return dataframe


def build_sensor_features(
    telemetry_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build leakage-safe historical sensor features.

    For each component, rolling features use the current observation
    plus historical observations only.

    The synthetic dataset contains daily observations, so:
    - 7 observations correspond to the Databricks 7-day window
    - 30 observations correspond to the Databricks 30-day window
    - shift(7) mirrors the Spark lag(..., 7) feature
    """
    dataframe = telemetry_df.copy()

    dataframe = dataframe.sort_values(
        [
            "component_id",
            "timestamp",
        ]
    ).reset_index(drop=True)

    grouped = dataframe.groupby(
        "component_id",
        sort=False,
    )

    dataframe["temperature_mean_7d"] = (
        grouped["temperature_c"]
        .transform(
            lambda series: series.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    dataframe["temperature_mean_30d"] = (
        grouped["temperature_c"]
        .transform(
            lambda series: series.rolling(
                window=30,
                min_periods=1,
            ).mean()
        )
    )

    dataframe["temperature_std_7d"] = (
        grouped["temperature_c"]
        .transform(
            lambda series: series.rolling(
                window=7,
                min_periods=2,
            ).std(ddof=1)
        )
    )

    dataframe["vibration_mean_7d"] = (
        grouped["vibration_mm_s"]
        .transform(
            lambda series: series.rolling(
                window=7,
                min_periods=1,
            ).mean()
        )
    )

    dataframe["vibration_mean_30d"] = (
        grouped["vibration_mm_s"]
        .transform(
            lambda series: series.rolling(
                window=30,
                min_periods=1,
            ).mean()
        )
    )

    dataframe["vibration_std_7d"] = (
        grouped["vibration_mm_s"]
        .transform(
            lambda series: series.rolling(
                window=7,
                min_periods=2,
            ).std(ddof=1)
        )
    )

    dataframe["temperature_delta_7d"] = (
        dataframe["temperature_c"]
        - grouped["temperature_c"].shift(7)
    )

    dataframe["vibration_delta_7d"] = (
        dataframe["vibration_mm_s"]
        - grouped["vibration_mm_s"].shift(7)
    )

    return dataframe


def add_component_features(
    telemetry_df: pd.DataFrame,
    components_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add observable component lifecycle information.

    susceptibility_factor is intentionally excluded because it is a
    hidden simulation-only characteristic.
    """
    required_columns = [
        "component_id",
        "expected_life_cycles",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in components_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Component data is missing required columns: "
            + ", ".join(missing_columns)
        )

    component_features = components_df[
        required_columns
    ].copy()

    dataframe = telemetry_df.merge(
        component_features,
        on="component_id",
        how="left",
        validate="many_to_one",
    )

    dataframe["cycle_life_ratio"] = (
        dataframe["flight_cycles"]
        / dataframe["expected_life_cycles"]
    )

    return dataframe


def add_aircraft_features(
    telemetry_df: pd.DataFrame,
    aircraft_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add observable aircraft-level operational context.
    """
    required_columns = [
        "aircraft_id",
        "aircraft_type",
        "age_years",
        "utilization_factor",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in aircraft_df.columns
    ]

    if missing_columns:
        raise ValueError(
            "Aircraft data is missing required columns: "
            + ", ".join(missing_columns)
        )

    aircraft_features = aircraft_df[
        required_columns
    ].copy()

    return telemetry_df.merge(
        aircraft_features,
        on="aircraft_id",
        how="left",
        validate="many_to_one",
    )


def validate_model_features(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate the local inference feature contract.
    """
    missing_columns = [
        column
        for column in MODEL_FEATURE_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Local feature table is missing model features: "
            + ", ".join(missing_columns)
        )

    required_non_null = [
        "aircraft_type",
        "component_type",
        "age_years",
        "utilization_factor",
        "flight_cycles",
        "expected_life_cycles",
        "cycle_life_ratio",
        "days_since_maintenance",
        "previous_failures",
        "vibration_mm_s",
        "temperature_missing_flag",
    ]

    null_counts = (
        dataframe[required_non_null]
        .isna()
        .sum()
    )

    invalid_columns = null_counts[
        null_counts > 0
    ]

    if not invalid_columns.empty:
        raise ValueError(
            "Unexpected null values in required local features: "
            + ", ".join(
                invalid_columns.index.tolist()
            )
        )


def build_local_feature_table() -> pd.DataFrame:
    """
    Build the local feature table used by AeroMaintain inference.

    This intentionally does not create the future-failure training
    target. Local inference only requires information available at or
    before the observation timestamp.
    """
    (
        telemetry_df,
        components_df,
        aircraft_df,
    ) = load_raw_data()

    feature_df = prepare_telemetry(
        telemetry_df
    )

    feature_df = build_sensor_features(
        feature_df
    )

    feature_df = add_component_features(
        feature_df,
        components_df,
    )

    feature_df = add_aircraft_features(
        feature_df,
        aircraft_df,
    )

    validate_model_features(
        feature_df
    )

    return feature_df


def get_latest_component_features(
    component_id: str,
) -> dict:
    """
    Return the most recent model feature record for one component.
    """
    if not component_id.strip():
        raise ValueError(
            "component_id must not be empty."
        )

    feature_df = build_local_feature_table()

    component_df = feature_df[
        feature_df["component_id"] == component_id
    ].copy()

    if component_df.empty:
        raise ValueError(
            f"Unknown component_id: {component_id}"
        )

    latest_row = (
        component_df
        .sort_values(
            "timestamp",
            ascending=False,
        )
        .iloc[0]
    )

    features = {
        column: latest_row[column]
        for column in MODEL_FEATURE_COLUMNS
    }

    return features


if __name__ == "__main__":
    feature_table = build_local_feature_table()

    print(
        "Local predictive feature table built successfully."
    )

    print(
        "Rows:",
        f"{len(feature_table):,}",
    )

    print(
        "Components:",
        feature_table[
            "component_id"
        ].nunique(),
    )

    print(
        "Model features:",
        len(MODEL_FEATURE_COLUMNS),
    )

    print(
        "Latest timestamp:",
        feature_table[
            "timestamp"
        ].max(),
    )