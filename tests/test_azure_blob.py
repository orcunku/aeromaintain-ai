"""Tests for the Azure Blob Storage integration."""

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from aeromaintain.cloud import azure_blob


def test_required_raw_datasets_match_project_contract():
    assert azure_blob.REQUIRED_RAW_DATASETS == (
        "aircraft.parquet",
        "components.parquet",
        "maintenance.parquet",
        "telemetry.parquet",
    )


def test_default_container_name():
    assert azure_blob.DEFAULT_CONTAINER_NAME == "aeromaintain-raw"


def test_missing_connection_string_raises_clear_error(monkeypatch):
    monkeypatch.delenv("AZURE_STORAGE_CONNECTION_STRING", raising=False)

    with pytest.raises(
        RuntimeError,
        match="AZURE_STORAGE_CONNECTION_STRING is not set",
    ):
        azure_blob._get_blob_service_client()


def test_blob_service_client_uses_environment_credential(
    monkeypatch,
):
    monkeypatch.setenv(
        "AZURE_STORAGE_CONNECTION_STRING",
        "synthetic-test-connection-string",
    )

    fake_client = MagicMock()

    monkeypatch.setattr(
        azure_blob.BlobServiceClient,
        "from_connection_string",
        MagicMock(return_value=fake_client),
    )

    result = azure_blob._get_blob_service_client()

    assert result is fake_client

    azure_blob.BlobServiceClient.from_connection_string.assert_called_once_with(
        "synthetic-test-connection-string"
    )


def test_list_raw_datasets_returns_sorted_blob_names(
    monkeypatch,
):
    fake_blobs = [
        MagicMock(name="telemetry.parquet"),
        MagicMock(name="aircraft.parquet"),
        MagicMock(name="maintenance.parquet"),
    ]

    for blob, name in zip(
        fake_blobs,
        (
            "telemetry.parquet",
            "aircraft.parquet",
            "maintenance.parquet",
        ),
    ):
        blob.name = name

    fake_container = MagicMock()
    fake_container.list_blobs.return_value = fake_blobs

    fake_service = MagicMock()
    fake_service.get_container_client.return_value = fake_container

    monkeypatch.setattr(
        azure_blob,
        "_get_blob_service_client",
        lambda: fake_service,
    )

    result = azure_blob.list_raw_datasets()

    assert result == [
        "aircraft.parquet",
        "maintenance.parquet",
        "telemetry.parquet",
    ]

    fake_service.get_container_client.assert_called_once_with(
        "aeromaintain-raw"
    )


def test_download_rejects_unknown_dataset(tmp_path):
    with pytest.raises(
        ValueError,
        match="Unsupported raw dataset",
    ):
        azure_blob.download_raw_dataset(
            "secret.txt",
            tmp_path,
        )


def test_download_raw_dataset_writes_blob_to_destination(
    monkeypatch,
    tmp_path,
):
    payload = b"synthetic parquet bytes"

    fake_download = MagicMock()

    def readinto(file_object):
        file_object.write(payload)

    fake_download.readinto.side_effect = readinto

    fake_blob_client = MagicMock()
    fake_blob_client.download_blob.return_value = fake_download

    fake_service = MagicMock()
    fake_service.get_blob_client.return_value = fake_blob_client

    monkeypatch.setattr(
        azure_blob,
        "_get_blob_service_client",
        lambda: fake_service,
    )

    result = azure_blob.download_raw_dataset(
        "aircraft.parquet",
        tmp_path,
    )

    assert result == Path(tmp_path) / "aircraft.parquet"
    assert result.exists()
    assert result.read_bytes() == payload

    fake_service.get_blob_client.assert_called_once_with(
        container="aeromaintain-raw",
        blob="aircraft.parquet",
    )


def test_download_creates_destination_directory(
    monkeypatch,
    tmp_path,
):
    destination = tmp_path / "nested" / "azure-data"

    fake_download = MagicMock()

    def readinto(file_object):
        file_object.write(b"data")

    fake_download.readinto.side_effect = readinto

    fake_blob_client = MagicMock()
    fake_blob_client.download_blob.return_value = fake_download

    fake_service = MagicMock()
    fake_service.get_blob_client.return_value = fake_blob_client

    monkeypatch.setattr(
        azure_blob,
        "_get_blob_service_client",
        lambda: fake_service,
    )

    result = azure_blob.download_raw_dataset(
        "components.parquet",
        destination,
    )

    assert destination.is_dir()
    assert result.exists()