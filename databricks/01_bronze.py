"""
AeroMaintain AI - Bronze Layer Ingestion

Reads synthetic raw Parquet files from a Unity Catalog Volume
and persists them as Delta tables.

Bronze principles:
- Preserve source data without cleaning.
- Add ingestion metadata for traceability.
- Store datasets as Delta tables in Unity Catalog.

Synthetic data only. Not real aircraft operational data.
"""

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, current_timestamp


CATALOG = "workspace"
SCHEMA = "aeromaintain"

RAW_VOLUME_PATH = (
    f"/Volumes/{CATALOG}/{SCHEMA}/raw_data"
)


def read_bronze_source(
    spark: SparkSession,
    file_name: str,
) -> DataFrame:
    """
    Read a raw Parquet source and add ingestion metadata.
    """

    source_path = f"{RAW_VOLUME_PATH}/{file_name}"

    return (
        spark.read
        .parquet(source_path)
        .withColumn(
            "_ingested_at",
            current_timestamp(),
        )
        .withColumn(
            "_source_file",
            col("_metadata.file_path"),
        )
    )


def write_delta_table(
    dataframe: DataFrame,
    table_name: str,
) -> None:
    """
    Persist a Spark DataFrame as a managed Delta table.
    """

    full_table_name = (
        f"{CATALOG}.{SCHEMA}.{table_name}"
    )

    (
        dataframe.write
        .format("delta")
        .mode("overwrite")
        .saveAsTable(full_table_name)
    )

    print(
        f"Created Delta table: {full_table_name}"
    )


def main() -> None:
    """
    Run Bronze ingestion for all AeroMaintain datasets.
    """

    spark = SparkSession.getActiveSession()

    if spark is None:
        raise RuntimeError(
            "No active Spark session found. "
            "Run this script inside Databricks."
        )

    sources = {
        "bronze_aircraft": "aircraft.parquet",
        "bronze_components": "components.parquet",
        "bronze_telemetry": "telemetry.parquet",
        "bronze_maintenance": "maintenance.parquet",
    }

    for table_name, file_name in sources.items():

        print(
            f"Ingesting {file_name} "
            f"-> {table_name}"
        )

        dataframe = read_bronze_source(
            spark=spark,
            file_name=file_name,
        )

        write_delta_table(
            dataframe=dataframe,
            table_name=table_name,
        )


if __name__ == "__main__":
    main()