"""Azure Blob Storage integration for AeroMaintain AI.

This module provides a small, explicit cloud-storage boundary for the
synthetic raw datasets used by the project.

Authentication is provided through the
AZURE_STORAGE_CONNECTION_STRING environment variable.

Credentials are intentionally never stored in source code.
"""

from __future__ import annotations

import os
from pathlib import Path

from azure.storage.blob import BlobServiceClient


DEFAULT_CONTAINER_NAME = "aeromaintain-raw"

REQUIRED_RAW_DATASETS = (
    "aircraft.parquet",
    "components.parquet",
    "maintenance.parquet",
    "telemetry.parquet",
)


def _get_blob_service_client() -> BlobServiceClient:
    """Create an authenticated Azure Blob service client.

    Raises
    ------
    RuntimeError
        If the Azure connection string is not available.
    """
    connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")

    if not connection_string:
        raise RuntimeError(
            "AZURE_STORAGE_CONNECTION_STRING is not set. "
            "Azure Blob Storage access is unavailable."
        )

    return BlobServiceClient.from_connection_string(connection_string)


def list_raw_datasets(
    container_name: str = DEFAULT_CONTAINER_NAME,
) -> list[str]:
    """Return blob names available in the raw-data container."""
    service_client = _get_blob_service_client()
    container_client = service_client.get_container_client(container_name)

    return sorted(blob.name for blob in container_client.list_blobs())


def download_raw_dataset(
    dataset_name: str,
    destination_dir: str | Path,
    container_name: str = DEFAULT_CONTAINER_NAME,
) -> Path:
    """Download one approved raw dataset from Azure Blob Storage.

    Parameters
    ----------
    dataset_name:
        Name of the raw Parquet dataset to download.
    destination_dir:
        Local directory where the dataset should be written.
    container_name:
        Azure Blob container containing the raw datasets.

    Returns
    -------
    pathlib.Path
        Path to the downloaded local file.

    Raises
    ------
    ValueError
        If the requested dataset is outside the approved raw-data contract.
    """
    if dataset_name not in REQUIRED_RAW_DATASETS:
        allowed = ", ".join(REQUIRED_RAW_DATASETS)
        raise ValueError(
            f"Unsupported raw dataset: {dataset_name!r}. "
            f"Expected one of: {allowed}."
        )

    destination_dir = Path(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)

    destination_path = destination_dir / dataset_name

    service_client = _get_blob_service_client()
    blob_client = service_client.get_blob_client(
        container=container_name,
        blob=dataset_name,
    )

    with destination_path.open("wb") as file:
        stream = blob_client.download_blob()
        stream.readinto(file)

    return destination_path