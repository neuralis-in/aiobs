"""Google Cloud Storage exporter for aiobs observability data.

Exports observability data to Google Cloud Storage buckets.
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Optional

from .base import BaseExporter, ExportResult, ExportError
from ..models import ObservabilityExport


class GCSExporter(BaseExporter):
    """Export observability data to Google Cloud Storage.

    Example usage:
        from aiobs.exporters import GCSExporter

        exporter = GCSExporter(
            bucket="my-observability-bucket",
            prefix="traces/",
            project="my-gcp-project",
        )
        observer.flush(exporter=exporter)

    Authentication:
        The exporter uses Google Cloud's default authentication chain:
        1. GOOGLE_APPLICATION_CREDENTIALS environment variable
        2. Service account credentials file path
        3. Application Default Credentials (ADC)

    Args:
        bucket: The GCS bucket name (required).
        prefix: Path prefix within the bucket (e.g., "traces/"). Defaults to "".
        project: GCP project ID. If not provided, uses the default project.
        credentials_path: Path to service account JSON file. If not provided,
                         uses default authentication.
        filename_template: Template for the output filename. Supports placeholders:
                          - {session_id}: First session ID
                          - {timestamp}: Unix timestamp
                          - {date}: Date in YYYY-MM-DD format
                          Defaults to "{session_id}.json"
        content_type: Content-Type for uploaded files. Defaults to "application/json".
    """

    name: str = "gcs"

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        project: Optional[str] = None,
        credentials_path: Optional[str] = None,
        filename_template: str = "{session_id}.json",
        content_type: str = "application/json",
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.rstrip("/") + "/" if prefix and not prefix.endswith("/") else prefix
        self.project = project
        self.credentials_path = credentials_path
        self.filename_template = filename_template
        self.content_type = content_type
        self._client: Any = None

    def _get_client(self) -> Any:
        """Lazily initialize the GCS client."""
        if self._client is None:
            try:
                from google.cloud import storage
                from google.oauth2 import service_account
            except ImportError as e:
                raise ExportError(
                    "google-cloud-storage is required for GCSExporter. "
                    "Install it with: pip install google-cloud-storage",
                    cause=e,
                )

            if self.credentials_path:
                credentials = service_account.Credentials.from_service_account_file(
                    self.credentials_path
                )
                self._client = storage.Client(
                    project=self.project,
                    credentials=credentials,
                )
            else:
                self._client = storage.Client(project=self.project)

        return self._client

    def _generate_filename(self, data: ObservabilityExport) -> str:
        """Generate the filename from the template."""
        session_id = "unknown"
        if data.sessions:
            session_id = data.sessions[0].id

        timestamp = int(time.time())
        date = time.strftime("%Y-%m-%d")

        return self.filename_template.format(
            session_id=session_id,
            timestamp=timestamp,
            date=date,
        )

    def export(self, data: ObservabilityExport, **kwargs: Any) -> ExportResult:
        """Export observability data to GCS.

        Args:
            data: The ObservabilityExport object to export.
            **kwargs: Additional options:
                - filename: Override the filename (ignores template).
                - metadata: Dict of custom metadata to attach to the blob.

        Returns:
            ExportResult with the GCS URI and export metadata.

        Raises:
            ExportError: If upload fails.
        """
        self.validate(data)

        try:
            client = self._get_client()
            bucket = client.bucket(self.bucket)

            # Generate filename
            filename = kwargs.get("filename") or self._generate_filename(data)
            blob_path = f"{self.prefix}{filename}"
            blob = bucket.blob(blob_path)

            # Serialize data
            json_data = json.dumps(data.model_dump(), ensure_ascii=False, indent=2)
            bytes_data = json_data.encode("utf-8")

            # Set custom metadata if provided
            if metadata := kwargs.get("metadata"):
                blob.metadata = metadata

            # Upload
            blob.upload_from_string(
                bytes_data,
                content_type=self.content_type,
            )

            gcs_uri = f"gs://{self.bucket}/{blob_path}"

            return ExportResult(
                success=True,
                destination=gcs_uri,
                bytes_written=len(bytes_data),
                metadata={
                    "bucket": self.bucket,
                    "blob_path": blob_path,
                    "content_type": self.content_type,
                    "sessions_count": len(data.sessions),
                    "events_count": len(data.events),
                    "function_events_count": len(data.function_events),
                },
            )

        except ExportError:
            raise
        except Exception as e:
            raise ExportError(f"Failed to export to GCS: {e}", cause=e)

    @classmethod
    def from_env(
        cls,
        bucket_env: str = "AIOBS_GCS_BUCKET",
        prefix_env: str = "AIOBS_GCS_PREFIX",
        project_env: str = "AIOBS_GCS_PROJECT",
        credentials_env: str = "GOOGLE_APPLICATION_CREDENTIALS",
    ) -> "GCSExporter":
        """Create a GCSExporter from environment variables.

        Args:
            bucket_env: Environment variable name for bucket.
            prefix_env: Environment variable name for prefix.
            project_env: Environment variable name for project.
            credentials_env: Environment variable name for credentials path.

        Returns:
            Configured GCSExporter instance.

        Raises:
            ExportError: If required environment variables are missing.
        """
        bucket = os.getenv(bucket_env)
        if not bucket:
            raise ExportError(f"Environment variable {bucket_env} is required")

        return cls(
            bucket=bucket,
            prefix=os.getenv(prefix_env, ""),
            project=os.getenv(project_env),
            credentials_path=os.getenv(credentials_env),
        )




