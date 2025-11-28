"""Base exporter interface for aiobs observability data.

Exporters handle the serialization and transport of observability data
to various destinations (files, cloud storage, databases, etc.).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

from ..models import ObservabilityExport


class BaseExporter(ABC):
    """Abstract base class for observability data exporters.

    Subclasses must implement the `export()` method to handle
    the actual data export to their specific destination.

    Example usage:
        class MyExporter(BaseExporter):
            def export(self, data: ObservabilityExport, **kwargs) -> ExportResult:
                # Custom export logic
                ...
    """

    name: str = "base"

    @abstractmethod
    def export(self, data: ObservabilityExport, **kwargs: Any) -> "ExportResult":
        """Export observability data to the destination.

        Args:
            data: The ObservabilityExport object containing all sessions,
                  events, and trace data.
            **kwargs: Additional exporter-specific options.

        Returns:
            ExportResult with status and metadata about the export.

        Raises:
            ExportError: If the export fails.
        """
        raise NotImplementedError

    def validate(self, data: ObservabilityExport) -> bool:
        """Validate data before export. Override for custom validation.

        Args:
            data: The data to validate.

        Returns:
            True if valid, raises ExportError otherwise.
        """
        if not data.sessions:
            return True  # Empty data is valid, just nothing to export
        return True


class ExportResult:
    """Result of an export operation.

    Attributes:
        success: Whether the export succeeded.
        destination: The destination where data was exported (URL, path, etc.).
        bytes_written: Number of bytes written (if applicable).
        metadata: Additional metadata about the export.
        error: Error message if export failed.
    """

    def __init__(
        self,
        success: bool,
        destination: Optional[str] = None,
        bytes_written: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        self.success = success
        self.destination = destination
        self.bytes_written = bytes_written
        self.metadata = metadata or {}
        self.error = error

    def __repr__(self) -> str:
        if self.success:
            return f"ExportResult(success=True, destination={self.destination!r}, bytes={self.bytes_written})"
        return f"ExportResult(success=False, error={self.error!r})"


class ExportError(Exception):
    """Exception raised when an export operation fails."""

    def __init__(self, message: str, cause: Optional[Exception] = None) -> None:
        super().__init__(message)
        self.cause = cause




