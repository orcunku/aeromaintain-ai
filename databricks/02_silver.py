"""
AeroMaintain AI - Silver Layer Transformation

Transforms Bronze Delta tables into validated Silver tables.

Silver responsibilities:
- Remove exact telemetry source duplicates.
- Preserve missing sensor values while adding data-quality flags.
- Detect vibration outliers using the IQR rule.
- Validate primary keys and referential integrity.
- Preserve lineage metadata from the Bronze layer.

Synthetic data only. Not real aircraft operational data.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, when


CATALOG = "workspace"
SCHEMA = "aeromaintain"


TELEMETRY_SOURCE_COLUMNS = [
    "timestamp",
    "aircraft_id",
    "component_id",
    "component_type",
    "flight_cycles",
    "days_since_maintenance",
    "previous_failures",
    "health_index",
    "temperature_c",
    "vibration_mm_s",
    "failure_event",
]


def table_name(name: str) -> str:
    """Return a fully qualified Unity Catalog table name."""

    return f"{CATALOG}.{SCHEMA}.{name}"


def validate_unique_key(
    dataframe: DataFrame,
    key_column: str,
    dataset_name: str,
) -> None:
    """Validate that a key column is non-null and unique."""

    total_rows = dataframe.count()

    unique_keys = (
        dataframe
        .select(key_column)
        .distinct()
        .count()
    )

    null_keys = (
        dataframe
        .filter(col(key_column).isNull())
        .count()
    )

    if null_keys > 0:
        raise ValueError(
            f"{dataset_name}: "
            f"{null_keys} null {key_column} values found."
        )

    if unique_keys != total_rows:
        raise ValueError(
            f"{dataset_name}: {key_column} is not unique. "
            f"Rows={total_rows}, unique={unique_keys}."
        )


def validate_reference(
    child_df: DataFrame,
    parent_df: DataFrame,
    key_column: str,
    dataset_name: str,
) -> None:
    """Validate that child keys exist in the parent dataset."""

    unknown_count = (
        child_df
        .join(
            parent_df.select(key_column).distinct(),
            on=key_column,
            how="left_anti",
        )
        .count()
    )

    if unknown_count > 0:
        raise ValueError(
            f"{dataset_name}: "
            f"{unknown_count} unknown {key_column} values found."
        )


def build_silver_telemetry(
    bronze_telemetry_df: DataFrame,
) -> DataFrame:
    """
    Deduplicate telemetry and add data-quality flags.

    Missing temperatures are intentionally preserved.
    Vibration outliers are flagged rather than removed.
    """

    silver_df = (
        bronze_telemetry_df
        .dropDuplicates(TELEMETRY_SOURCE_COLUMNS)
    )

    q1, q3 = silver_df.approxQuantile(
        "vibration_mm_s",
        [0.25, 0.75],
        0.001,
    )

    iqr = q3 - q1
    vibration_upper_bound = q3 + (1.5 * iqr)

    silver_df = (
        silver_df
        .withColumn(
            "temperature_missing_flag",
            when(
                col("temperature_c").isNull(),
                1,
            ).otherwise(0),
        )
        .withColumn(
            "vibration_outlier_flag",
            when(
                col("vibration_mm_s")
                > vibration_upper_bound,
                1,
            ).otherwise(0),
        )
    )

    print(
        "Vibration IQR upper bound: "
        f"{vibration_upper_bound:.3f}"
    )

    return silver_df


def write_delta_table(
    dataframe: DataFrame,
    name: str,
) -> None:
    """Write a DataFrame as a managed Silver Delta table."""

    full_name = table_name(name)

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(full_name)
    )

    print(f"Created Delta table: {full_name}")


def main() -> None:
    """Run the AeroMaintain Silver transformation pipeline."""

    spark = SparkSession.getActiveSession()

    if spark is None:
        raise RuntimeError(
            "No active Spark session found. "
            "Run this script inside Databricks."
        )

    aircraft_df = spark.table(
        table_name("bronze_aircraft")
    )

    components_df = spark.table(
        table_name("bronze_components")
    )

    telemetry_df = spark.table(
        table_name("bronze_telemetry")
    )

    maintenance_df = spark.table(
        table_name("bronze_maintenance")
    )

    # Validate master-data primary keys.
    validate_unique_key(
        aircraft_df,
        "aircraft_id",
        "aircraft",
    )

    validate_unique_key(
        components_df,
        "component_id",
        "components",
    )

    validate_unique_key(
        maintenance_df,
        "work_order_id",
        "maintenance",
    )

    # Validate relationships between datasets.
    validate_reference(
        components_df,
        aircraft_df,
        "aircraft_id",
        "components",
    )

    validate_reference(
        maintenance_df,
        aircraft_df,
        "aircraft_id",
        "maintenance",
    )

    validate_reference(
        maintenance_df,
        components_df,
        "component_id",
        "maintenance",
    )

    # Build Silver telemetry.
    silver_telemetry_df = build_silver_telemetry(
        telemetry_df
    )

    # Aircraft, components, and maintenance passed
    # validation and require no value transformations.
    write_delta_table(
        aircraft_df,
        "silver_aircraft",
    )

    write_delta_table(
        components_df,
        "silver_components",
    )

    write_delta_table(
        silver_telemetry_df,
        "silver_telemetry",
    )

    write_delta_table(
        maintenance_df,
        "silver_maintenance",
    )


if __name__ == "__main__":
    main()