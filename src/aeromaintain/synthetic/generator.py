from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from aeromaintain.config import RAW_DATA_DIR, RANDOM_SEED


# ---------------------------------------------------------------------
# SYNTHETIC SIMULATION ASSUMPTIONS
# ---------------------------------------------------------------------
#
# These values exist only to create a realistic engineering/ML
# simulation. They are NOT real aircraft or OEM specifications.
# ---------------------------------------------------------------------

COMPONENT_TYPES = {
    "HYDRAULIC_PUMP": {
        "base_temp": 65.0,
        "base_vibration": 2.0,
        "expected_life_cycles": 4500,
        "degradation_factor": 1.20,
        "failure_factor": 1.25,
    },
    "GENERATOR": {
        "base_temp": 75.0,
        "base_vibration": 1.8,
        "expected_life_cycles": 6000,
        "degradation_factor": 1.00,
        "failure_factor": 1.00,
    },
    "FUEL_PUMP": {
        "base_temp": 55.0,
        "base_vibration": 1.5,
        "expected_life_cycles": 5000,
        "degradation_factor": 1.10,
        "failure_factor": 1.10,
    },
    "ACTUATOR": {
        "base_temp": 50.0,
        "base_vibration": 1.2,
        "expected_life_cycles": 7000,
        "degradation_factor": 0.85,
        "failure_factor": 0.80,
    },
    "AIR_CYCLE_MACHINE": {
        "base_temp": 80.0,
        "base_vibration": 2.4,
        "expected_life_cycles": 5500,
        "degradation_factor": 1.15,
        "failure_factor": 1.20,
    },
}


def create_fleet(
    n_aircraft: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Create the synthetic aircraft fleet.

    utilization_factor represents how intensively an aircraft
    operates relative to the simulated fleet.
    """

    records = []

    aircraft_types = [
        "A320_SIM",
        "B737_SIM",
        "E190_SIM",
    ]

    for i in range(n_aircraft):

        age_years = round(
            float(rng.uniform(1, 18)),
            1,
        )

        utilization_factor = float(
            np.clip(
                rng.normal(1.0, 0.15),
                0.65,
                1.40,
            )
        )

        records.append(
            {
                "aircraft_id": f"AC-{i + 1:03d}",
                "aircraft_type": rng.choice(
                    aircraft_types
                ),
                "age_years": age_years,
                "entry_into_service_year":
                    datetime.now().year - int(age_years),
                "utilization_factor":
                    round(utilization_factor, 3),
            }
        )

    return pd.DataFrame(records)


def create_components(
    fleet: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Assign one component of each simulated type
    to every aircraft.

    susceptibility_factor is a hidden characteristic.
    The future ML model will NOT receive this feature.

    This prevents the synthetic ML problem from becoming
    completely deterministic.
    """

    records = []

    component_counter = 1

    for aircraft_id in fleet["aircraft_id"]:

        for component_type, specs in COMPONENT_TYPES.items():

            initial_cycles = int(
                rng.uniform(
                    0,
                    specs["expected_life_cycles"] * 0.75,
                )
            )

            susceptibility_factor = float(
                np.clip(
                    rng.lognormal(
                        mean=0.0,
                        sigma=0.18,
                    ),
                    0.70,
                    1.50,
                )
            )

            records.append(
                {
                    "component_id":
                        f"CMP-{component_counter:05d}",
                    "aircraft_id":
                        aircraft_id,
                    "component_type":
                        component_type,
                    "cycles_at_start":
                        initial_cycles,
                    "expected_life_cycles":
                        specs["expected_life_cycles"],
                    "susceptibility_factor":
                        round(
                            susceptibility_factor,
                            3,
                        ),
                }
            )

            component_counter += 1

    return pd.DataFrame(records)


def simulate_operations(
    fleet: pd.DataFrame,
    components: pd.DataFrame,
    days: int,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Simulate daily aircraft operations.

    The simulation creates relationships between:

    utilization
    -> cycles
    -> degradation
    -> sensor deterioration
    -> failure probability
    -> maintenance
    """

    telemetry_records = []
    maintenance_records = []

    start_date = datetime(2025, 1, 1)

    maintenance_counter = 1

    component_state = {}

    for component in components.itertuples():

        component_state[component.component_id] = {
            "cycles": component.cycles_at_start,
            "health": float(
                rng.uniform(0.90, 1.0)
            ),
            "days_since_maintenance": int(
                rng.integers(0, 90)
            ),
            "previous_failures": 0,
        }

    # Faster lookup than filtering the full dataframe
    # repeatedly inside the daily simulation.
    component_groups = {
        aircraft_id: group
        for aircraft_id, group
        in components.groupby("aircraft_id")
    }

    for day in range(days):

        current_date = (
            start_date
            + timedelta(days=day)
        )

        for aircraft in fleet.itertuples():

            # Base operations for this aircraft today.
            base_flights = int(
                rng.integers(1, 6)
            )

            flights_today = max(
                1,
                int(
                    round(
                        base_flights
                        * aircraft.utilization_factor
                    )
                ),
            )

            aircraft_components = (
                component_groups[
                    aircraft.aircraft_id
                ]
            )

            for component in (
                aircraft_components.itertuples()
            ):

                state = component_state[
                    component.component_id
                ]

                specs = COMPONENT_TYPES[
                    component.component_type
                ]

                state["cycles"] += flights_today

                state[
                    "days_since_maintenance"
                ] += 1

                age_ratio = (
                    state["cycles"]
                    / component.expected_life_cycles
                )

                # Older components degrade faster,
                # especially beyond ~70% expected life.
                age_acceleration = (
                    1.0
                    + max(
                        0.0,
                        age_ratio - 0.70,
                    ) * 2.5
                )

                utilization_effect = (
                    0.90
                    + 0.10
                    * aircraft.utilization_factor
                )

                degradation_rate = (
                    0.000045
                    * specs["degradation_factor"]
                    * component.susceptibility_factor
                    * age_acceleration
                    * utilization_effect
                )

                degradation_noise = float(
                    rng.uniform(
                        0.00000,
                        0.000035,
                    )
                )

                state["health"] -= (
                    degradation_rate
                    + degradation_noise
                )

                state["health"] = float(
                    np.clip(
                        state["health"],
                        0.0,
                        1.0,
                    )
                )

                degradation = (
                    1.0
                    - state["health"]
                )

                # -------------------------------------------------
                # SENSOR GENERATION
                # -------------------------------------------------

                temperature = (
                    specs["base_temp"]
                    + degradation * 55
                    + max(
                        0.0,
                        age_ratio - 0.75,
                    ) * 8
                    + rng.normal(
                        0,
                        1.8,
                    )
                )

                vibration = (
                    specs["base_vibration"]
                    + degradation * 8
                    + max(
                        0.0,
                        age_ratio - 0.75,
                    ) * 1.5
                    + rng.normal(
                        0,
                        0.20,
                    )
                )

                # -------------------------------------------------
                # FAILURE MODEL
                # -------------------------------------------------

                health_risk = max(
                    0.0,
                    0.94 - state["health"],
                )

                age_risk = max(
                    0.0,
                    age_ratio - 0.65,
                )

                repeat_failure_risk = min(
                    state["previous_failures"]
                    * 0.0004,
                    0.003,
                )

                failure_probability = (
                    0.00020
                    + health_risk * 0.035
                    + age_risk * 0.010
                    + repeat_failure_risk
                )

                failure_probability *= (
                    specs["failure_factor"]
                    * component.susceptibility_factor
                )

                failure_probability = float(
                    np.clip(
                        failure_probability,
                        0.00005,
                        0.08,
                    )
                )

                failure_event = (
                    rng.random()
                    < failure_probability
                )

                telemetry_records.append(
                    {
                        "timestamp":
                            current_date,
                        "aircraft_id":
                            aircraft.aircraft_id,
                        "component_id":
                            component.component_id,
                        "component_type":
                            component.component_type,
                        "flight_cycles":
                            state["cycles"],
                        "days_since_maintenance":
                            state[
                                "days_since_maintenance"
                            ],
                        "previous_failures":
                            state[
                                "previous_failures"
                            ],
                        "health_index":
                            round(
                                state["health"],
                                4,
                            ),
                        "temperature_c":
                            round(
                                temperature,
                                2,
                            ),
                        "vibration_mm_s":
                            round(
                                vibration,
                                3,
                            ),
                        "failure_event":
                            int(
                                failure_event
                            ),
                    }
                )

                # -------------------------------------------------
                # MAINTENANCE
                # -------------------------------------------------

                if failure_event:

                    state[
                        "previous_failures"
                    ] += 1

                    action = rng.choice(
                        [
                            "INSPECTED",
                            "REPAIRED",
                            "REPLACED",
                        ],
                        p=[
                            0.20,
                            0.45,
                            0.35,
                        ],
                    )

                    maintenance_records.append(
                        {
                            "work_order_id":
                                (
                                    "WO-"
                                    f"{maintenance_counter:06d}"
                                ),
                            "timestamp":
                                current_date,
                            "aircraft_id":
                                aircraft.aircraft_id,
                            "component_id":
                                component.component_id,
                            "component_type":
                                component.component_type,
                            "event_type":
                                "CORRECTIVE",
                            "fault_code":
                                (
                                    "FLT-"
                                    f"{rng.integers(100, 999)}"
                                ),
                            "technician_note":
                                (
                                    "Inspection performed "
                                    "following abnormal "
                                    f"{component.component_type} "
                                    "condition."
                                ),
                            "corrective_action":
                                action,
                        }
                    )

                    maintenance_counter += 1

                    # Different maintenance actions
                    # have different effects.

                    if action == "INSPECTED":

                        state["health"] = min(
                            1.0,
                            state["health"]
                            + float(
                                rng.uniform(
                                    0.01,
                                    0.03,
                                )
                            ),
                        )

                    elif action == "REPAIRED":

                        state["health"] = min(
                            1.0,
                            state["health"]
                            + float(
                                rng.uniform(
                                    0.05,
                                    0.12,
                                )
                            ),
                        )

                    elif action == "REPLACED":

                        state["health"] = float(
                            rng.uniform(
                                0.96,
                                1.0,
                            )
                        )

                        # Replacement means component
                        # effective operating age is reset.
                        state["cycles"] = int(
                            rng.integers(
                                0,
                                50,
                            )
                        )

                    state[
                        "days_since_maintenance"
                    ] = 0

    return (
        pd.DataFrame(
            telemetry_records
        ),
        pd.DataFrame(
            maintenance_records
        ),
    )


def inject_data_quality_issues(
    telemetry: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """
    Inject controlled data-quality problems.

    These will later be detected and handled
    by the Databricks Silver layer.
    """

    df = telemetry.copy()

    # Missing temperature values.
    missing_count = max(
        1,
        int(
            len(df) * 0.005
        ),
    )

    missing_indices = rng.choice(
        df.index,
        size=missing_count,
        replace=False,
    )

    df.loc[
        missing_indices,
        "temperature_c",
    ] = np.nan

    # Extreme vibration outliers.
    outlier_count = max(
        1,
        int(
            len(df) * 0.001
        ),
    )

    outlier_indices = rng.choice(
        df.index,
        size=outlier_count,
        replace=False,
    )

    df.loc[
        outlier_indices,
        "vibration_mm_s",
    ] *= 10

    # Duplicate records.
    duplicates = df.sample(
        frac=0.002,
        random_state=RANDOM_SEED,
    )

    df = pd.concat(
        [
            df,
            duplicates,
        ],
        ignore_index=True,
    )

    return df


def generate_dataset(
    n_aircraft: int = 100,
    days: int = 365,
    output_dir: Path = RAW_DATA_DIR,
) -> None:

    rng = np.random.default_rng(
        RANDOM_SEED
    )

    print(
        "1/4 Generating fleet..."
    )

    fleet = create_fleet(
        n_aircraft=n_aircraft,
        rng=rng,
    )

    print(
        "2/4 Generating components..."
    )

    components = create_components(
        fleet=fleet,
        rng=rng,
    )

    print(
        "3/4 Simulating operations..."
    )

    telemetry, maintenance = (
        simulate_operations(
            fleet=fleet,
            components=components,
            days=days,
            rng=rng,
        )
    )

    print(
        "4/4 Injecting data-quality problems..."
    )

    telemetry_dirty = (
        inject_data_quality_issues(
            telemetry=telemetry,
            rng=rng,
        )
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    fleet.to_parquet(
        output_dir
        / "aircraft.parquet",
        index=False,
    )

    components.to_parquet(
        output_dir
        / "components.parquet",
        index=False,
    )

    telemetry_dirty.to_parquet(
        output_dir
        / "telemetry.parquet",
        index=False,
    )

    maintenance.to_parquet(
        output_dir
        / "maintenance.parquet",
        index=False,
    )

    print(
        "\nAeroMaintain synthetic dataset created."
    )

    print(
        "-" * 45
    )

    print(
        f"Aircraft:           "
        f"{len(fleet):,}"
    )

    print(
        f"Components:         "
        f"{len(components):,}"
    )

    print(
        f"Telemetry records:  "
        f"{len(telemetry_dirty):,}"
    )

    print(
        f"Maintenance events: "
        f"{len(maintenance):,}"
    )

    print(
        f"\nSaved to: "
        f"{output_dir}"
    )


if __name__ == "__main__":

    generate_dataset()