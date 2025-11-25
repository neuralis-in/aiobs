from __future__ import annotations

import inspect
import os
import time
from typing import Any, Callable, Dict, Optional

from .base_api import BaseGeminiAPIModule
from .models import (
    GenerateContentRequest,
    GenerateContentResponse,
)
from ....models import Event as ObsEvent, Callsite as ObsCallsite


class GenerateContentAPI(BaseGeminiAPIModule):
    name = "models.generate_content"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from google.genai.models import Models  # noqa: F401
            return hasattr(Models, "generate_content")
        except Exception:
            return False

    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        try:
            from google.genai.models import Models as _Models  # type: ignore
        except Exception:
            return None

        if not hasattr(_Models, "generate_content"):
            return None

        original_generate_content = _Models.generate_content

        def _extract_request(args: tuple, kwargs: dict) -> GenerateContentRequest:
            req: Dict[str, Any] = {}
            # generate_content signature: generate_content(self, *, model, contents, config=None, ...)
            # args[0] is self (the Models instance)
            if args:
                req["_args"] = args[1:]  # Skip self
            req.update(kwargs or {})
            
            model = req.get("model")
            contents = req.get("contents")
            system_instruction = req.get("system_instruction")
            config = req.get("config")
            
            # Convert config to dict if it's a pydantic model or has model_dump
            config_dict = None
            if config is not None:
                try:
                    if hasattr(config, "model_dump"):
                        config_dict = config.model_dump()
                    elif hasattr(config, "dict"):
                        config_dict = config.dict()
                    elif isinstance(config, dict):
                        config_dict = config
                except Exception:
                    config_dict = str(config)
            
            # Simplify contents for storage
            contents_simplified = None
            if contents is not None:
                try:
                    if isinstance(contents, str):
                        contents_simplified = contents
                    elif isinstance(contents, list):
                        # Truncate to first 3 items for preview
                        preview = contents[:3]
                        contents_simplified = []
                        for item in preview:
                            if hasattr(item, "model_dump"):
                                contents_simplified.append(item.model_dump())
                            elif hasattr(item, "dict"):
                                contents_simplified.append(item.dict())
                            elif isinstance(item, dict):
                                contents_simplified.append(item)
                            else:
                                contents_simplified.append(str(item))
                    else:
                        contents_simplified = str(contents)
                except Exception:
                    contents_simplified = str(contents)
            
            return GenerateContentRequest(
                model=model,
                contents=contents_simplified,
                system_instruction=system_instruction,
                config=config_dict,
                other={k: v for k, v in req.items() 
                       if k not in {"model", "contents", "system_instruction", "config", "_args"}},
            )

        def _extract_response(resp: Any) -> GenerateContentResponse:
            rmodel = getattr(resp, "model", None)
            text = None
            candidates = None
            usage = None
            
            # Extract text from response
            try:
                # Gemini response has .text property for simple text extraction
                if hasattr(resp, "text"):
                    text = resp.text
                elif hasattr(resp, "candidates") and resp.candidates:
                    # Fallback: extract from first candidate
                    first_candidate = resp.candidates[0]
                    if hasattr(first_candidate, "content"):
                        content = first_candidate.content
                        if hasattr(content, "parts") and content.parts:
                            first_part = content.parts[0]
                            if hasattr(first_part, "text"):
                                text = first_part.text
            except Exception:
                pass
            
            # Extract candidates info
            try:
                if hasattr(resp, "candidates") and resp.candidates:
                    candidates = []
                    for cand in resp.candidates:
                        cand_dict = {}
                        if hasattr(cand, "model_dump"):
                            cand_dict = cand.model_dump()
                        elif hasattr(cand, "dict"):
                            cand_dict = cand.dict()
                        candidates.append(cand_dict)
            except Exception:
                pass
            
            # Extract usage metadata
            try:
                usage_meta = getattr(resp, "usage_metadata", None)
                if usage_meta is not None:
                    if hasattr(usage_meta, "model_dump"):
                        usage = usage_meta.model_dump()
                    elif hasattr(usage_meta, "dict"):
                        usage = usage_meta.dict()
                    elif isinstance(usage_meta, dict):
                        usage = usage_meta
                    else:
                        # Try to extract common attributes
                        usage = {}
                        for attr in ["prompt_token_count", "candidates_token_count", "total_token_count"]:
                            if hasattr(usage_meta, attr):
                                usage[attr] = getattr(usage_meta, attr)
            except Exception:
                pass
            
            return GenerateContentResponse(
                model=rmodel,
                text=text,
                candidates=candidates,
                usage=usage,
            )

        def _get_callsite() -> Optional[ObsCallsite]:
            try:
                frames = inspect.stack()[2:]
                for fi in frames:
                    fname = os.path.abspath(fi.filename)
                    if (
                        f"{os.sep}aiobs{os.sep}" in fname
                        or f"{os.sep}site-packages{os.sep}google{os.sep}" in fname
                        or "google/genai" in fname
                    ):
                        continue
                    try:
                        rel = os.path.relpath(fname, start=os.getcwd())
                    except Exception:
                        rel = fname
                    return ObsCallsite(file=rel, line=fi.lineno, function=fi.function)
            except Exception:
                pass
            return None

        def wrapped(self, *args, **kwargs):  # type: ignore[no-redef]
            import uuid
            started = time.time()
            request_info = _extract_request((self,) + args, kwargs)
            callsite = _get_callsite()
            error: Optional[str] = None
            response_info: Optional[GenerateContentResponse] = None
            
            # Capture parent span context for trace tree linking
            parent_span_id = collector.get_current_span_id()
            span_id = str(uuid.uuid4())
            
            try:
                resp = original_generate_content(self, *args, **kwargs)
                response_info = _extract_response(resp)
                return resp
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                ended = time.time()
                event = ObsEvent(
                    provider="gemini",
                    api="models.generate_content",
                    callsite=callsite,
                    request=request_info,
                    response=response_info,
                    error=error,
                    started_at=started,
                    ended_at=ended,
                    duration_ms=round((ended - started) * 1000, 3),
                    span_id=span_id,
                    parent_span_id=parent_span_id,
                )
                collector._record_event(event)

        setattr(_Models, "generate_content", wrapped)  # type: ignore[attr-defined]

        def unpatch() -> None:
            try:
                setattr(_Models, "generate_content", original_generate_content)  # type: ignore[attr-defined]
            except Exception:
                pass

        return unpatch

