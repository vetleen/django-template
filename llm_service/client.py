"""
Entry points: completion(**kwargs), acompletion(**kwargs), get_client().
Applies policy (timeout, retry, guardrails), logs to LLMCallLog, proxies to BaseLLMClient.
"""
import asyncio
import hashlib
import logging
import time
from typing import Any, AsyncIterator, Iterator

from django.db import connection

from llm_service.base import BaseLLMClient
from llm_service.conf import (
    get_default_model,
    get_log_write_timeout,
    get_max_retries,
    get_post_call_hooks,
    get_pre_call_hooks,
    is_model_allowed,
)
from llm_service.litellm_client import LiteLLMClient
from llm_service.models import LLMCallLog
from llm_service.pricing import get_fallback_cost_usd
from llm_service.request_result import LLMRequest, LLMResult

logger = logging.getLogger(__name__)

# Module-level client instance (lazy)
_client: BaseLLMClient | None = None


def get_client() -> BaseLLMClient:
    """Return the configured LLM client (LiteLLM by default)."""
    global _client
    if _client is None:
        _client = LiteLLMClient()
    return _client


def _kwargs_to_request(**kwargs: Any) -> LLMRequest:
    """Build LLMRequest from completion(**kwargs)."""
    model = kwargs.pop("model", None) or get_default_model()
    messages = kwargs.get("messages", [])
    stream = kwargs.get("stream", False)
    metadata = kwargs.pop("metadata", None) or {}
    user = kwargs.pop("user", None)
    request_id = kwargs.pop("request_id", None)
    if request_id is not None and "request_id" not in metadata:
        metadata = {**metadata, "request_id": str(request_id)}
    return LLMRequest(
        model=model,
        messages=messages,
        stream=stream,
        metadata=metadata,
        raw_kwargs=kwargs,
        user=user,
    )


def _response_to_result(response: Any, model: str) -> LLMResult:
    """Build LLMResult from LiteLLM completion response."""
    usage = {}
    if getattr(response, "usage", None):
        u = response.usage
        usage = {
            "input_tokens": getattr(u, "prompt_tokens", 0) or getattr(u, "input_tokens", 0),
            "output_tokens": getattr(u, "completion_tokens", 0) or getattr(u, "output_tokens", 0),
            "total_tokens": getattr(u, "total_tokens", 0),
        }
    cost = None
    try:
        hidden = getattr(response, "_hidden_params", None) or {}
        cost = hidden.get("response_cost")
    except Exception:
        pass
    if cost is None and usage:
        fallback = get_fallback_cost_usd(
            model,
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
        )
        if fallback is not None:
            cost = float(fallback)
    text = None
    if getattr(response, "choices", None) and len(response.choices) > 0:
        c = response.choices[0]
        if getattr(c, "message", None) and getattr(c.message, "content", None):
            text = c.message.content
    return LLMResult(
        text=text,
        usage=usage or None,
        cost=cost,
        raw_response=response,
        provider_response_id=getattr(response, "id", None),
        response_model=getattr(response, "model", None),
    )


def _truncate(s: str, max_len: int = 2000) -> str:
    if not s or len(s) <= max_len:
        return s or ""
    return s[:max_len] + "..."

def _hash_preview(s: str, max_len: int = 64) -> str:
    if not s:
        return ""
    return hashlib.sha256(s.encode("utf-8", errors="replace")).hexdigest()[:max_len]


def _sanitize_kwargs(kwargs: dict) -> dict:
    """Remove or redact sensitive keys; truncate large values."""
    out = {}
    skip = {"api_key", "api_key_id", "credentials"}
    for k, v in kwargs.items():
        if k.lower() in skip:
            continue
        if isinstance(v, str) and len(v) > 500:
            out[k] = _truncate(v, 500)
        elif isinstance(v, (list, dict)) and len(str(v)) > 1000:
            out[k] = "<truncated>"
        else:
            out[k] = v
    return out


def _write_log(
    request: LLMRequest,
    result: LLMResult,
    duration_ms: int | None,
    status: str = LLMCallLog.Status.SUCCESS,
    error_type: str | None = None,
    error_message: str = "",
    http_status: int | None = None,
    retry_count: int = 0,
) -> LLMCallLog | None:
    """Write full LLMCallLog. Returns None on failure (caller may try minimal)."""
    try:
        prompt_preview = ""
        if request.messages:
            parts = []
            for m in request.messages[:5]:
                content = m.get("content") if isinstance(m.get("content"), str) else str(m.get("content", ""))[:500]
                parts.append(content)
            prompt_preview = _truncate("\n".join(parts), 2000)
        request_id = (request.metadata or {}).get("request_id", "")
        from decimal import Decimal
        cost_usd = None
        cost_source = None
        if result.cost is not None:
            cost_usd = Decimal(str(result.cost))
            cost_source = "litellm"
        else:
            fb = get_fallback_cost_usd(request.model, result.input_tokens, result.output_tokens)
            if fb is not None:
                cost_usd = fb
                cost_source = "fallback"
        log = LLMCallLog.objects.create(
            model=request.model,
            is_stream=request.stream,
            user=request.user,
            metadata=request.metadata or {},
            request_id=request_id,
            duration_ms=duration_ms,
            request_kwargs=_sanitize_kwargs(request.raw_kwargs),
            prompt_hash=_hash_preview(prompt_preview),
            prompt_preview=prompt_preview,
            provider_response_id=result.provider_response_id,
            response_model=result.response_model,
            response_preview=_truncate(result.text or "", 2000),
            response_hash=_hash_preview(result.text or ""),
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            total_tokens=result.total_tokens,
            cost_usd=cost_usd,
            cost_source=cost_source,
            status=status,
            error_type=error_type,
            error_message=_truncate(error_message, 2000),
            http_status=http_status,
            retry_count=retry_count,
        )
        return log
    except Exception as e:
        logger.warning("LLMCallLog full write failed: %s", e)
        return None


def _write_minimal_log(request: LLMRequest, note: str) -> None:
    """Fallback: minimal row with primitives only when full log fails."""
    try:
        LLMCallLog.objects.create(
            model=request.model or "unknown",
            is_stream=request.stream,
            status=LLMCallLog.Status.LOGGING_FAILED,
            error_message=_truncate(note, 500),
        )
    except Exception as e:
        logger.warning("LLMCallLog minimal write failed: %s", e)


def _run_pre_hooks(request: LLMRequest) -> None:
    for hook in get_pre_call_hooks():
        try:
            hook(request)
        except Exception as e:
            logger.info("Pre-call hook blocked: %s", e)
            raise


def _run_post_hooks(result: LLMResult) -> None:
    for hook in get_post_call_hooks():
        try:
            hook(result)
        except Exception as e:
            logger.info("Post-call hook blocked: %s", e)
            raise


def _save_log_with_timeout(request: LLMRequest, result: LLMResult, duration_ms: int | None, status: str = LLMCallLog.Status.SUCCESS, error_type: str | None = None, error_message: str = "", http_status: int | None = None, retry_count: int = 0) -> None:
    """Write log with DB timeout; on failure write minimal row."""
    timeout = get_log_write_timeout()
    try:
        connection.set_parameter("statement_timeout", int(timeout * 1000))
    except Exception:
        pass
    try:
        log = _write_log(request, result, duration_ms, status=status, error_type=error_type, error_message=error_message, http_status=http_status, retry_count=retry_count)
        if log is None:
            _write_minimal_log(request, "log write failed")
    except Exception as e:
        _write_minimal_log(request, f"log write failed: {e}")
    finally:
        try:
            connection.set_parameter("statement_timeout", 0)
        except Exception:
            pass


def _is_retryable(e: Exception) -> bool:
    s = str(e).lower()
    if "429" in s or "rate limit" in s or "timeout" in s:
        return True
    if "503" in s or "502" in s or "500" in s:
        return True
    return False


def completion(**kwargs: Any) -> Any:
    """
    Sync completion. Validates model, runs pre/post hooks, retries, logs.
    With stream=True returns an iterator that proxies provider chunks and writes LLMCallLog on exit.
    """
    request = _kwargs_to_request(**kwargs)
    if not is_model_allowed(request.model):
        from llm_service.conf import get_allowed_models
        raise ValueError(f"Model not allowed: {request.model}. Allowed: {get_allowed_models()}")
    _run_pre_hooks(request)
    client = get_client()
    start = time.perf_counter()
    last_error = None
    retry_count = 0
    max_retries = get_max_retries()
    for attempt in range(max_retries + 1):
        try:
            if request.stream:
                return _stream_sync(request, client, start)
            resp = client.completion(**request.to_completion_kwargs())
            duration_ms = int((time.perf_counter() - start) * 1000)
            result = _response_to_result(resp, request.model)
            _run_post_hooks(result)
            _save_log_with_timeout(request, result, duration_ms)
            return resp
        except Exception as e:
            last_error = e
            retry_count = attempt
            if attempt < max_retries and _is_retryable(e):
                time.sleep(min(2 ** attempt, 60))
                continue
            duration_ms = int((time.perf_counter() - start) * 1000)
            result = LLMResult(error=e, usage={}, text=None)
            _save_log_with_timeout(
                request, result, duration_ms,
                status=LLMCallLog.Status.ERROR,
                error_type=type(e).__name__,
                error_message=str(e),
                retry_count=retry_count,
            )
            raise
    raise last_error


def _stream_sync(request: LLMRequest, client: BaseLLMClient, start: float) -> Iterator[Any]:
    """Wrap stream so we finalize and write LLMCallLog when iterator ends or is closed."""
    result_holder: list[LLMResult] = []
    usage_holder: list[dict] = []
    final_response_holder: list[Any] = []
    try:
        stream = client.completion(**request.to_completion_kwargs())
        for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_holder.append({
                    "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or getattr(chunk.usage, "input_tokens", 0),
                    "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or getattr(chunk.usage, "output_tokens", 0),
                    "total_tokens": getattr(chunk.usage, "total_tokens", 0),
                })
            if getattr(chunk, "choices", None) and len(chunk.choices) > 0 and getattr(chunk.choices[0], "delta", None):
                pass
            final_response_holder.append(chunk)
            yield chunk
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        u = usage_holder[-1] if usage_holder else {}
        result = LLMResult(error=e, usage=u, text=None)
        _save_log_with_timeout(request, result, duration_ms, status=LLMCallLog.Status.ERROR, error_type=type(e).__name__, error_message=str(e))
        raise
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        u = usage_holder[-1] if usage_holder else {}
        cost = None
        try:
            last = final_response_holder[-1] if final_response_holder else None
            if last and getattr(last, "_hidden_params", None):
                cost = last._hidden_params.get("response_cost")
        except Exception:
            pass
        if cost is None and u:
            cost = get_fallback_cost_usd(request.model, u.get("input_tokens", 0), u.get("output_tokens", 0))
            cost = float(cost) if cost is not None else None
        result = LLMResult(usage=u, cost=cost, text=None)
        _save_log_with_timeout(request, result, duration_ms)


async def acompletion(**kwargs: Any) -> Any:
    """
    Async completion. Same policy and logging as completion().
    With stream=True returns an async iterator.
    """
    request = _kwargs_to_request(**kwargs)
    if not is_model_allowed(request.model):
        from llm_service.conf import get_allowed_models
        raise ValueError(f"Model not allowed: {request.model}. Allowed: {get_allowed_models()}")
    _run_pre_hooks(request)
    client = get_client()
    start = time.perf_counter()
    last_error = None
    retry_count = 0
    max_retries = get_max_retries()
    for attempt in range(max_retries + 1):
        try:
            if request.stream:
                return _stream_async(request, client, start)
            resp = await client.acompletion(**request.to_completion_kwargs())
            duration_ms = int((time.perf_counter() - start) * 1000)
            result = _response_to_result(resp, request.model)
            _run_post_hooks(result)
            _save_log_with_timeout(request, result, duration_ms)
            return resp
        except Exception as e:
            last_error = e
            retry_count = attempt
            if attempt < max_retries and _is_retryable(e):
                await asyncio.sleep(min(2 ** attempt, 60))
                continue
            duration_ms = int((time.perf_counter() - start) * 1000)
            result = LLMResult(error=e, usage={}, text=None)
            _save_log_with_timeout(request, result, duration_ms, status=LLMCallLog.Status.ERROR, error_type=type(e).__name__, error_message=str(e), retry_count=retry_count)
            raise
    raise last_error


async def _stream_async(request: LLMRequest, client: BaseLLMClient, start: float) -> AsyncIterator[Any]:
    usage_holder: list[dict] = []
    final_response_holder: list[Any] = []
    try:
        stream = await client.acompletion(**request.to_completion_kwargs())
        async for chunk in stream:
            if getattr(chunk, "usage", None):
                usage_holder.append({
                    "input_tokens": getattr(chunk.usage, "prompt_tokens", 0) or getattr(chunk.usage, "input_tokens", 0),
                    "output_tokens": getattr(chunk.usage, "completion_tokens", 0) or getattr(chunk.usage, "output_tokens", 0),
                    "total_tokens": getattr(chunk.usage, "total_tokens", 0),
                })
            final_response_holder.append(chunk)
            yield chunk
    except Exception as e:
        duration_ms = int((time.perf_counter() - start) * 1000)
        u = usage_holder[-1] if usage_holder else {}
        result = LLMResult(error=e, usage=u, text=None)
        _save_log_with_timeout(request, result, duration_ms, status=LLMCallLog.Status.ERROR, error_type=type(e).__name__, error_message=str(e))
        raise
    finally:
        duration_ms = int((time.perf_counter() - start) * 1000)
        u = usage_holder[-1] if usage_holder else {}
        cost = None
        try:
            last = final_response_holder[-1] if final_response_holder else None
            if last and getattr(last, "_hidden_params", None):
                cost = last._hidden_params.get("response_cost")
        except Exception:
            pass
        if cost is None and u:
            fb = get_fallback_cost_usd(request.model, u.get("input_tokens", 0), u.get("output_tokens", 0))
            cost = float(fb) if fb is not None else None
        result = LLMResult(usage=u, cost=cost, text=None)
        _save_log_with_timeout(request, result, duration_ms)