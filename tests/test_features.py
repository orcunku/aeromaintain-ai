import numpy as np
import pandas as pd
import pytest

from aeromaintain.features.build_features import (
    MODEL_FEATURE_COLUMNS,
    add_aircraft_features,
    add_component_features,
    build_sensor_features,
    get_latest_component_features,
    prepare_telemetry,
    validate_model_features,
)


def build_sample_telemetry() -> pd.DataFrame:
    """
    Build a small deterministic telemetry dataset for feature tests.
    """
    timestamps = pd.date_range(
        "2025-01-01",
        periods=8,
        freq="D",
    )

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "aircraft_id": ["AC-001"] * 8,
            "component_id": ["CMP-001"] * 8,
            "component_type": ["HYDRAULIC_PUMP"] * 8,
            "flight_cycles": [
                100,
                110,
                120,
                130,
                140,
                150,
                160,
                170,
            ],
            "days_since_maintenance": list(range(8)),
            "previous_failures": [0] * 8,
            "temperature_c": [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
                110.0,
            ],
            "vibration_mm_s": [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
                2.0,
            ],
        }
    )


def test_prepare_telemetry_removes_duplicates_and_flags_missing():
    """
    Duplicate component/timestamp observations should be removed and
    originally missing temperatures should receive a missing flag.
    """
    telemetry = pd.DataFrame(
        {
            "timestamp": [
                "2025-01-01",
                "2025-01-01",
                "2025-01-02",
            ],
            "component_id": [
                "CMP-001",
                "CMP-001",
                "CMP-001",
            ],
            "temperature_c": [
                100.0,
                999.0,
                np.nan,
            ],
        }
    )

    result = prepare_telemetry(telemetry)

    assert len(result) == 2

    assert (
        result.duplicated(
            subset=["component_id", "timestamp"]
        ).sum()
        == 0
    )

    assert result["temperature_missing_flag"].tolist() == [
        0,
        1,
    ]

    assert pd.isna(
        result.loc[
            result["temperature_missing_flag"] == 1,
            "temperature_c",
        ].iloc[0]
    )


def test_build_sensor_features_uses_historical_rolling_windows():
    """
    Rolling sensor features should use only the current and previous
    observations for each component.
    """
    telemetry = build_sample_telemetry()

    result = build_sensor_features(telemetry)

    seventh_row = result.iloc[6]

    assert seventh_row["temperature_mean_7d"] == pytest.approx(
        np.mean(
            [
                100.0,
                101.0,
                102.0,
                103.0,
                104.0,
                105.0,
                106.0,
            ]
        )
    )

    assert seventh_row["vibration_mean_7d"] == pytest.approx(
        np.mean(
            [
                1.0,
                1.1,
                1.2,
                1.3,
                1.4,
                1.5,
                1.6,
            ]
        )
    )


def test_build_sensor_features_calculates_seven_day_delta():
    """
    The eighth observation should compare against the observation
    seven rows earlier for the same component.
    """
    telemetry = build_sample_telemetry()

    result = build_sensor_features(telemetry)

    eighth_row = result.iloc[7]

    assert eighth_row["temperature_delta_7d"] == pytest.approx(
        10.0
    )

    assert eighth_row["vibration_delta_7d"] == pytest.approx(
        1.0
    )


def test_sensor_features_do_not_mix_components():
    """
    Rolling calculations must remain isolated within each component.
    """
    telemetry = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                [
                    "2025-01-01",
                    "2025-01-02",
                    "2025-01-01",
                    "2025-01-02",
                ]
            ),
            "component_id": [
                "CMP-001",
                "CMP-001",
                "CMP-002",
                "CMP-002",
            ],
            "temperature_c": [
                10.0,
                20.0,
                100.0,
                200.0,
            ],
            "vibration_mm_s": [
                1.0,
                2.0,
                10.0,
                20.0,
            ],
        }
    )

    result = build_sensor_features(telemetry)

    cmp_001 = result[
        result["component_id"] == "CMP-001"
    ].reset_index(drop=True)

    cmp_002 = result[
        result["component_id"] == "CMP-002"
    ].reset_index(drop=True)

    assert cmp_001.loc[1, "temperature_mean_7d"] == pytest.approx(
        15.0
    )

    assert cmp_002.loc[1, "temperature_mean_7d"] == pytest.approx(
        150.0
    )

    assert cmp_001.loc[1, "vibration_mean_7d"] == pytest.approx(
        1.5
    )

    assert cmp_002.loc[1, "vibration_mean_7d"] == pytest.approx(
        15.0
    )


def test_add_component_features_calculates_cycle_life_ratio():
    """
    Component metadata should add expected life and derive the
    observable cycle-life ratio.
    """
    telemetry = pd.DataFrame(
        {
            "component_id": ["CMP-001"],
            "flight_cycles": [500],
        }
    )

    components = pd.DataFrame(
        {
            "component_id": ["CMP-001"],
            "expected_life_cycles": [1000],
            "susceptibility_factor": [1.25],
        }
    )

    result = add_component_features(
        telemetry,
        components,
    )

    assert result.loc[0, "expected_life_cycles"] == 1000

    assert result.loc[0, "cycle_life_ratio"] == pytest.approx(
        0.5
    )

    assert "susceptibility_factor" not in result.columns


def test_add_component_features_requires_expected_life():
    """
    Component metadata without expected life should fail explicitly.
    """
    telemetry = pd.DataFrame(
        {
            "component_id": ["CMP-001"],
            "flight_cycles": [500],
        }
    )

    components = pd.DataFrame(
        {
            "component_id": ["CMP-001"],
        }
    )

    with pytest.raises(
        ValueError,
        match="expected_life_cycles",
    ):
        add_component_features(
            telemetry,
            components,
        )


def test_add_aircraft_features_adds_operational_context():
    """
    Aircraft metadata should add observable fleet context used by the
    predictive model.
    """
    telemetry = pd.DataFrame(
        {
            "aircraft_id": ["AC-001"],
            "component_id": ["CMP-001"],
        }
    )

    aircraft = pd.DataFrame(
        {
            "aircraft_id": ["AC-001"],
            "aircraft_type": ["A320"],
            "age_years": [8.0],
            "utilization_factor": [1.1],
        }
    )

    result = add_aircraft_features(
        telemetry,
        aircraft,
    )

    assert result.loc[0, "aircraft_type"] == "A320"
    assert result.loc[0, "age_years"] == pytest.approx(8.0)
    assert result.loc[0, "utilization_factor"] == pytest.approx(
        1.1
    )


def test_validate_model_features_accepts_valid_contract():
    """
    A dataframe containing the complete model feature contract should
    pass validation.
    """
    row = {
        column: 1.0
        for column in MODEL_FEATURE_COLUMNS
    }

    row["aircraft_type"] = "A320"
    row["component_type"] = "HYDRAULIC_PUMP"

    dataframe = pd.DataFrame([row])

    validate_model_features(dataframe)


def test_validate_model_features_rejects_missing_feature():
    """
    Missing model features should fail with a clear validation error.
    """
    row = {
        column: 1.0
        for column in MODEL_FEATURE_COLUMNS
        if column != "cycle_life_ratio"
    }

    row["aircraft_type"] = "A320"
    row["component_type"] = "HYDRAULIC_PUMP"

    dataframe = pd.DataFrame([row])

    with pytest.raises(
        ValueError,
        match="cycle_life_ratio",
    ):
        validate_model_features(dataframe)


def test_validate_model_features_rejects_required_null():
    """
    Required operational features should not contain unexpected nulls.
    """
    row = {
        column: 1.0
        for column in MODEL_FEATURE_COLUMNS
    }

    row["aircraft_type"] = "A320"
    row["component_type"] = "HYDRAULIC_PUMP"
    row["flight_cycles"] = np.nan

    dataframe = pd.DataFrame([row])

    with pytest.raises(
        ValueError,
        match="flight_cycles",
    ):
        validate_model_features(dataframe)


def test_model_feature_contract_excludes_simulation_only_fields():
    """
    Hidden simulation state and target information must not appear in
    the production inference feature contract.
    """
    forbidden_features = {
        "health_index",
        "susceptibility_factor",
        "failure_event",
        "future_failure_count",
        "failure_next_30d",
    }

    assert forbidden_features.isdisjoint(
        MODEL_FEATURE_COLUMNS
    )


def test_get_latest_component_features_returns_model_contract(
    monkeypatch,
):
    """
    Latest-component inference should return exactly the model feature
    contract from the newest available observation.
    """
    older_row = {
        column: 1.0
        for column in MODEL_FEATURE_COLUMNS
    }

    newer_row = {
        column: 2.0
        for column in MODEL_FEATURE_COLUMNS
    }

    older_row["aircraft_type"] = "A320"
    older_row["component_type"] = "HYDRAULIC_PUMP"

    newer_row["aircraft_type"] = "A320"
    newer_row["component_type"] = "HYDRAULIC_PUMP"

    older_row.update(
        {
            "component_id": "CMP-001",
            "timestamp": pd.Timestamp("2025-01-01"),
        }
    )

    newer_row.update(
        {
            "component_id": "CMP-001",
            "timestamp": pd.Timestamp("2025-01-02"),
        }
    )

    fake_feature_table = pd.DataFrame(
        [
            older_row,
            newer_row,
        ]
    )

    monkeypatch.setattr(
        "aeromaintain.features.build_features."
        "build_local_feature_table",
        lambda: fake_feature_table,
    )

    result = get_latest_component_features(
        "CMP-001"
    )

    assert list(result.keys()) == MODEL_FEATURE_COLUMNS

    assert result["flight_cycles"] == pytest.approx(
        2.0
    )


def test_get_latest_component_features_rejects_unknown_component(
    monkeypatch,
):
    """
    Unknown component identifiers should fail explicitly.
    """
    row = {
        column: 1.0
        for column in MODEL_FEATURE_COLUMNS
    }

    row["aircraft_type"] = "A320"
    row["component_type"] = "HYDRAULIC_PUMP"
    row["component_id"] = "CMP-001"
    row["timestamp"] = pd.Timestamp("2025-01-01")

    fake_feature_table = pd.DataFrame([row])

    monkeypatch.setattr(
        "aeromaintain.features.build_features."
        "build_local_feature_table",
        lambda: fake_feature_table,
    )

    with pytest.raises(
        ValueError,
        match="Unknown component_id",
    ):
        get_latest_component_features(
            "CMP-UNKNOWN"
        )


def test_get_latest_component_features_rejects_empty_id():
    """
    Empty component identifiers should fail before feature-table work.
    """
    with pytest.raises(
        ValueError,
        match="component_id must not be empty",
    ):
        get_latest_component_features("   ")