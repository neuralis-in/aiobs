from __future__ import annotations

import inspect
import os
import time
from typing import Any, Callable, Dict, Optional

from .base_api import BaseOpenAIAPIModule
from .models import (
    ChatCompletionsRequest,
    ChatCompletionsResponse,
    Message,
)
from ....models import Event as ObsEvent, Callsite as ObsCallsite


class ChatCompletionsAPI(BaseOpenAIAPIModule):
    name = "chat.completions"

    @classmethod
    def is_available(cls) -> bool:
        try:
            from openai.resources.chat.completions import Completions as _  # noqa: F401
            return True
        except Exception:
            return False

    def install(self, collector: Any) -> Optional[Callable[[], None]]:
        try:
            from openai.resources.chat.completions import Completions as _Completions  # type: ignore
        except Exception:
            return None

        if not hasattr(_Completions, "create"):
            return None

        original_create = _Completions.create

        def _extract_request(args: tuple, kwargs: dict) -> ChatCompletionsRequest:
            req: Dict[str, Any] = {}
            if args:
                req["_args"] = args
            req.update(kwargs or {})
            messages = req.get("messages")
            if messages is not None:
                try:
                    preview = messages[:3]
                except Exception:
                    preview = messages
                req["messages"] = preview
            msgs = None
            if req.get("messages") is not None:
                try:
                    msgs = [Message(**m) if not isinstance(m, Message) else m for m in req["messages"]]
                except Exception:
                    # Fallback to raw data if shape is unexpected
                    msgs = req.get("messages")
            return ChatCompletionsRequest(
                model=req.get("model"),
                messages=msgs,  # type: ignore[arg-type]
                temperature=req.get("temperature"),
                max_tokens=req.get("max_tokens"),
                other={k: v for k, v in req.items() if k not in {"model", "messages", "temperature", "max_tokens"}},
            )

        def _extract_response(resp: Any) -> ChatCompletionsResponse:
            rid = getattr(resp, "id", None)
            rmodel = getattr(resp, "model", None)
            text = None
            try:
                choices = getattr(resp, "choices", None)
                if choices:
                    msg = choices[0].message if len(choices) > 0 else None
                    if isinstance(msg, dict):
                        text = msg.get("content")
                    else:
                        text = getattr(msg, "content", None)
            except Exception:
                pass
            usage = getattr(resp, "usage", None)
            u = None
            try:
                if usage is not None:
                    if hasattr(usage, "model_dump"):
                        u = usage.model_dump()
                    elif hasattr(usage, "dict"):
                        u = usage.dict()
                    elif isinstance(usage, dict):
                        u = usage
            except Exception:
                pass
            return ChatCompletionsResponse(id=rid, model=rmodel, text=text, usage=u)

        def _get_callsite() -> Optional[ObsCallsite]:
            try:
                frames = inspect.stack()[2:]
                for fi in frames:
                    fname = os.path.abspath(fi.filename)
                    if (
                        f"{os.sep}aiobs{os.sep}" in fname
                        or f"{os.sep}site-packages{os.sep}openai{os.sep}" in fname
                        or fname.endswith("openai/__init__.py")
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
            request_info = _extract_request(args, kwargs)
            callsite = _get_callsite()
            error: Optional[str] = None
            response_info: Optional[ChatCompletionsResponse] = None
            
            # Capture parent span context for trace tree linking
            parent_span_id = collector.get_current_span_id()
            span_id = str(uuid.uuid4())
            
            try:
                resp = original_create(self, *args, **kwargs)
                response_info = _extract_response(resp)
                return resp
            except Exception as e:  # noqa: BLE001
                error = f"{type(e).__name__}: {e}"
                raise
            finally:
                ended = time.time()
                event = ObsEvent(
                    provider="openai",
                    api="chat.completions.create",
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

        setattr(_Completions, "create", wrapped)  # type: ignore[attr-defined]

        def unpatch() -> None:
            try:
                setattr(_Completions, "create", original_create)  # type: ignore[attr-defined]
            except Exception:
                pass

        return unpatch
