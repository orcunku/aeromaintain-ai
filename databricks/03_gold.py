"""
AeroMaintain AI - Gold Predictive Feature Pipeline

Builds a leakage-aware predictive-maintenance feature table from
validated Silver Delta tables.

Gold responsibilities:
- Create a 30-day future-failure prediction target.
- Remove right-censored observations.
- Build historical sensor features using past/current data only.
- Add component lifecycle and aircraft operational context.
- Exclude simulation-only latent variables from the ML feature table.
- Persist the final feature dataset as a Delta table.

Synthetic data only. Not real aircraft operational data.
"""

from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql.functions import (
    avg,
    col,
    date_sub,
    lag,
    lit,
    max as spark_max,
    stddev_samp,
    sum as spark_sum,
    unix_timestamp,
    when,
)


CATALOG = "workspace"
SCHEMA = "aeromaintain"

SILVER_TELEMETRY = f"{CATALOG}.{SCHEMA}.silver_telemetry"
SILVER_COMPONENTS = f"{CATALOG}.{SCHEMA}.silver_components"
SILVER_AIRCRAFT = f"{CATALOG}.{SCHEMA}.silver_aircraft"

GOLD_PREDICTIVE_FEATURES = (
    f"{CATALOG}.{SCHEMA}.gold_predictive_features"
)

SECONDS_PER_DAY = 24 * 60 * 60
PREDICTION_HORIZON_DAYS = 30


def build_failure_target(
    telemetry_df: DataFrame,
) -> DataFrame:
    """
    Create a binary target indicating whether a component will
    experience at least one failure in the next 30 days.

    The current row is excluded from the future window.
    Observations without a complete 30-day future horizon are
    removed to prevent right-censoring from being labeled as
    non-failure.
    """

    future_failure_window = (
        Window
        .partitionBy("component_id")
        .orderBy(unix_timestamp("timestamp"))
        .rangeBetween(
            1,
            PREDICTION_HORIZON_DAYS * SECONDS_PER_DAY,
        )
    )

    labeled_df = (
        telemetry_df
        .withColumn(
            "failures_next_30_days",
            spark_sum("failure_event").over(
                future_failure_window
            ),
        )
    )

    max_timestamp = (
        telemetry_df
        .agg(spark_max("timestamp"))
        .first()[0]
    )

    label_cutoff_date = date_sub(
        lit(max_timestamp),
        PREDICTION_HORIZON_DAYS,
    )

    labeled_df = (
        labeled_df
        .filter(
            col("timestamp") <= label_cutoff_date
        )
        .withColumn(
            "failure_within_30_days",
            when(
                col("failures_next_30_days") >= 1,
                1,
            ).otherwise(0),
        )
    )

    return labeled_df


def build_sensor_features(
    dataframe: DataFrame,
) -> DataFrame:
    """
    Build leakage-safe historical sensor features.

    Rolling windows contain only the current observation and
    historical observations for the same component.
    """

    history_7d_window = (
        Window
        .partitionBy("component_id")
        .orderBy(unix_timestamp("timestamp"))
        .rangeBetween(
            -6 * SECONDS_PER_DAY,
            0,
        )
    )

    history_30d_window = (
        Window
        .partitionBy("component_id")
        .orderBy(unix_timestamp("timestamp"))
        .rangeBetween(
            -29 * SECONDS_PER_DAY,
            0,
        )
    )

    component_time_window = (
        Window
        .partitionBy("component_id")
        .orderBy("timestamp")
    )

    return (
        dataframe
        .withColumn(
            "vibration_mean_7d",
            avg("vibration_mm_s").over(
                history_7d_window
            ),
        )
        .withColumn(
            "vibration_mean_30d",
            avg("vibration_mm_s").over(
                history_30d_window
            ),
        )
        .withColumn(
            "temperature_mean_7d",
            avg("temperature_c").over(
                history_7d_window
            ),
        )
        .withColumn(
            "temperature_mean_30d",
            avg("temperature_c").over(
                history_30d_window
            ),
        )
        .withColumn(
            "vibration_std_7d",
            stddev_samp("vibration_mm_s").over(
                history_7d_window
            ),
        )
        .withColumn(
            "temperature_std_7d",
            stddev_samp("temperature_c").over(
                history_7d_window
            ),
        )
        .withColumn(
            "vibration_7d_ago",
            lag(
                "vibration_mm_s",
                7,
            ).over(component_time_window),
        )
        .withColumn(
            "vibration_delta_7d",
            col("vibration_mm_s")
            - col("vibration_7d_ago"),
        )
        .withColumn(
            "temperature_7d_ago",
            lag(
                "temperature_c",
                7,
            ).over(component_time_window),
        )
        .withColumn(
            "temperature_delta_7d",
            col("temperature_c")
            - col("temperature_7d_ago"),
        )
    )


def add_component_features(
    dataframe: DataFrame,
    components_df: DataFrame,
) -> DataFrame:
    """
    Add observable component lifecycle information.

    susceptibility_factor is intentionally excluded because it is
    a hidden simulation-only characteristic.
    """

    component_features_df = (
        components_df
        .select(
            "component_id",
            "expected_life_cycles",
        )
    )

    return (
        dataframe
        .join(
            component_features_df,
            on="component_id",
            how="left",
        )
        .withColumn(
            "cycle_life_ratio",
            col("flight_cycles")
            / col("expected_life_cycles"),
        )
    )


def add_aircraft_features(
    dataframe: DataFrame,
    aircraft_df: DataFrame,
) -> DataFrame:
    """
    Add observable aircraft-level operational context.
    """

    aircraft_features_df = (
        aircraft_df
        .select(
            "aircraft_id",
            "aircraft_type",
            "age_years",
            "utilization_factor",
        )
    )

    return dataframe.join(
        aircraft_features_df,
        on="aircraft_id",
        how="left",
    )


def select_final_features(
    dataframe: DataFrame,
) -> DataFrame:
    """
    Select the final Gold feature contract.

    Identifiers remain for traceability and downstream evaluation,
    but they should not be used directly as ML model features.

    Simulation-only latent variables and future-information columns
    are intentionally excluded.
    """

    final_columns = [
        # Traceability
        "timestamp",
        "aircraft_id",
        "component_id",

        # Categorical context
        "aircraft_type",
        "component_type",

        # Operational / lifecycle features
        "age_years",
        "utilization_factor",
        "flight_cycles",
        "expected_life_cycles",
        "cycle_life_ratio",
        "days_since_maintenance",
        "previous_failures",

        # Current sensor features
        "temperature_c",
        "vibration_mm_s",
        "temperature_missing_flag",
        "vibration_outlier_flag",

        # Historical sensor features
        "temperature_mean_7d",
        "temperature_mean_30d",
        "temperature_std_7d",
        "temperature_delta_7d",
        "vibration_mean_7d",
        "vibration_mean_30d",
        "vibration_std_7d",
        "vibration_delta_7d",

        # Prediction target
        "failure_within_30_days",
    ]

    return dataframe.select(final_columns)


def validate_gold(
    dataframe: DataFrame,
) -> None:
    """
    Run structural validation before persisting the Gold table.
    """

    total_rows = dataframe.count()

    unique_observations = (
        dataframe
        .select(
            "component_id",
            "timestamp",
        )
        .distinct()
        .count()
    )

    if total_rows != unique_observations:
        raise ValueError(
            "Gold grain validation failed: "
            f"rows={total_rows}, "
            f"unique component-date rows={unique_observations}."
        )

    required_non_null_columns = [
        "aircraft_id",
        "component_id",
        "timestamp",
        "aircraft_type",
        "component_type",
        "expected_life_cycles",
        "cycle_life_ratio",
        "utilization_factor",
        "age_years",
        "failure_within_30_days",
    ]

    for column_name in required_non_null_columns:
        null_count = (
            dataframe
            .filter(col(column_name).isNull())
            .count()
        )

        if null_count > 0:
            raise ValueError(
                f"Gold validation failed: "
                f"{column_name} contains "
                f"{null_count} null values."
            )

    print(
        "Gold validation passed:",
        f"{total_rows:,} rows",
    )


def write_gold_table(
    dataframe: DataFrame,
) -> None:
    """
    Persist the predictive feature dataset as a Delta table.
    """

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(GOLD_PREDICTIVE_FEATURES)
    )

    print(
        "Created Gold Delta table:",
        GOLD_PREDICTIVE_FEATURES,
    )


def main() -> None:
    spark = SparkSession.getActiveSession()

    if spark is None:
        raise RuntimeError(
            "No active Spark session found. "
            "Run this script inside Databricks."
        )

    telemetry_df = spark.table(
        SILVER_TELEMETRY
    )

    components_df = spark.table(
        SILVER_COMPONENTS
    )

    aircraft_df = spark.table(
        SILVER_AIRCRAFT
    )

    gold_df = build_failure_target(
        telemetry_df
    )

    gold_df = build_sensor_features(
        gold_df
    )

    gold_df = add_component_features(
        gold_df,
        components_df,
    )

    gold_df = add_aircraft_features(
        gold_df,
        aircraft_df,
    )

    gold_df = select_final_features(
        gold_df
    )

    validate_gold(
        gold_df
    )

    write_gold_table(
        gold_df
    )


if __name__ == "__main__":
    main()