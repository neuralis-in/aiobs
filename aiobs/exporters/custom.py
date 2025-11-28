"""Custom exporter for user-defined export logic.

Allows users to define their own export behavior via callbacks.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Union

from .base import BaseExporter, ExportResult, ExportError
from ..models import ObservabilityExport


# Type alias for the export handler function
ExportHandler = Callable[[ObservabilityExport, Dict[str, Any]], Union[ExportResult, Dict[str, Any], None]]


class CustomExporter(BaseExporter):
    """Export observability data using a user-defined handler function.

    This exporter allows maximum flexibility by letting users define
    their own export logic via a callback function.

    Example usage:
        from aiobs.exporters import CustomExporter, ExportResult

        def my_export_handler(data, options):
            # Send to custom API, database, etc.
            response = requests.post(
                "https://my-api.com/traces",
                json=data.model_dump(),
            )
            return ExportResult(
                success=response.ok,
                destination="https://my-api.com/traces",
                metadata={"status_code": response.status_code},
            )

        exporter = CustomExporter(handler=my_export_handler)
        observer.flush(exporter=exporter)

    Args:
        handler: A callable that takes (data: ObservabilityExport, options: Dict)
                and returns an ExportResult, a dict with result data, or None.
        name: Optional name for this exporter instance.
        default_options: Default options to pass to the handler.
    """

    name: str = "custom"

    def __init__(
        self,
        handler: ExportHandler,
        name: Optional[str] = None,
        default_options: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not callable(handler):
            raise ValueError("handler must be a callable")

        self._handler = handler
        if name:
            self.name = name
        self._default_options = default_options or {}

    def export(self, data: ObservabilityExport, **kwargs: Any) -> ExportResult:
        """Export using the custom handler.

        Args:
            data: The ObservabilityExport object to export.
            **kwargs: Additional options passed to the handler.

        Returns:
            ExportResult from the handler.

        Raises:
            ExportError: If the handler fails or returns invalid data.
        """
        self.validate(data)

        # Merge default options with kwargs
        options = {**self._default_options, **kwargs}

        try:
            result = self._handler(data, options)

            # Handle different return types
            if result is None:
                return ExportResult(
                    success=True,
                    metadata={"handler": self.name},
                )
            elif isinstance(result, ExportResult):
                return result
            elif isinstance(result, dict):
                return ExportResult(
                    success=result.get("success", True),
                    destination=result.get("destination"),
                    bytes_written=result.get("bytes_written"),
                    metadata=result.get("metadata", {}),
                    error=result.get("error"),
                )
            else:
                # Unexpected return type, treat as success
                return ExportResult(
                    success=True,
                    metadata={"handler": self.name, "result": str(result)},
                )

        except Exception as e:
            raise ExportError(f"Custom export handler failed: {e}", cause=e)


class CompositeExporter(BaseExporter):
    """Export to multiple destinations using multiple exporters.

    Runs multiple exporters in sequence, collecting results from each.

    Example usage:
        from aiobs.exporters import CompositeExporter, GCSExporter, CustomExporter

        gcs = GCSExporter(bucket="my-bucket", prefix="traces/")
        custom = CustomExporter(handler=my_handler)

        exporter = CompositeExporter([gcs, custom])
        observer.flush(exporter=exporter)

    Args:
        exporters: List of exporters to run.
        stop_on_error: If True, stop on first error. If False, continue and
                      collect all results. Defaults to False.
    """

    name: str = "composite"

    def __init__(
        self,
        exporters: list[BaseExporter],
        stop_on_error: bool = False,
    ) -> None:
        if not exporters:
            raise ValueError("At least one exporter is required")

        self._exporters = exporters
        self._stop_on_error = stop_on_error

    def export(self, data: ObservabilityExport, **kwargs: Any) -> ExportResult:
        """Export using all configured exporters.

        Args:
            data: The ObservabilityExport object to export.
            **kwargs: Additional options passed to each exporter.

        Returns:
            ExportResult with aggregated metadata from all exporters.

        Raises:
            ExportError: If stop_on_error is True and any exporter fails.
        """
        self.validate(data)

        results: list[Dict[str, Any]] = []
        errors: list[str] = []
        all_success = True

        for exporter in self._exporters:
            try:
                result = exporter.export(data, **kwargs)
                results.append({
                    "exporter": exporter.name,
                    "success": result.success,
                    "destination": result.destination,
                    "bytes_written": result.bytes_written,
                    "metadata": result.metadata,
                    "error": result.error,
                })
                if not result.success:
                    all_success = False
                    if result.error:
                        errors.append(f"{exporter.name}: {result.error}")
            except ExportError as e:
                all_success = False
                errors.append(f"{exporter.name}: {e}")
                results.append({
                    "exporter": exporter.name,
                    "success": False,
                    "error": str(e),
                })
                if self._stop_on_error:
                    raise

        return ExportResult(
            success=all_success,
            metadata={
                "exporters_count": len(self._exporters),
                "results": results,
            },
            error="; ".join(errors) if errors else None,
        )

    def add(self, exporter: BaseExporter) -> "CompositeExporter":
        """Add an exporter to the composite.

        Args:
            exporter: The exporter to add.

        Returns:
            Self for method chaining.
        """
        self._exporters.append(exporter)
        return self




