"""
LLM Service: thin router that delegates to providers (e.g. OpenAI).
Owns LLMCallLog creation and usage tracking; providers return neutral result dicts.
"""

import inspect
import logging

from decimal import Decimal
from datetime import datetime

from django.db import transaction

from llm_service.models import LLMCallLog, UserMonthlyUsage
from llm_service.providers import get_provider

logger = logging.getLogger(__name__)


def _get_caller_info():
    try:
        frame = inspect.stack()[2]
        return f"{frame.filename}:{frame.function}:{frame.lineno}"
    except (IndexError, AttributeError):
        return "unknown"


def _result_to_call_log(result: dict, caller_info: str) -> LLMCallLog:
    """Build and save LLMCallLog from a provider result dict."""
    return LLMCallLog.objects.create(
        raw_response=result["raw_response"],
        caller=caller_info,
        model=result["model"],
        reasoning_effort=result["reasoning_effort"],
        system_instructions=result["system_instructions"],
        user_prompt=result["user_prompt"],
        json_schema=result["json_schema"],
        schema_name=result["schema_name"],
        parsed_json=result.get("parsed_json"),
        input_tokens=result.get("input_tokens", 0),
        output_tokens=result.get("output_tokens", 0),
        cached_tokens=result.get("cached_tokens", 0),
        reasoning_tokens=result.get("reasoning_tokens", 0),
        total_tokens=result.get("total_tokens", 0),
        succeeded=result.get("succeeded", False),
        llm_cost_usd=result.get("llm_cost_usd", 0),
        response_time_seconds=result.get("response_time_seconds", 0),
    )


def _track_usage(user, cost, total_tokens: int):
    """Update UserMonthlyUsage for the given user."""
    now = datetime.now()
    with transaction.atomic():
        try:
            usage = UserMonthlyUsage.objects.select_for_update().get(
                user=user, year=now.year, month=now.month
            )
            usage.total_cost_usd += Decimal(str(cost))
            usage.total_tokens += total_tokens
            usage.total_calls += 1
            usage.save()
        except UserMonthlyUsage.DoesNotExist:
            UserMonthlyUsage.objects.create(
                user=user,
                year=now.year,
                month=now.month,
                total_cost_usd=Decimal(str(cost)),
                total_tokens=total_tokens,
                total_calls=1,
            )


class LLMService:
    """
    Central service for LLM operations. Routes to the appropriate provider
    (e.g. OpenAI) based on model, creates LLMCallLog, and tracks usage.
    """

    def __init__(self):
        self.model = "gpt-5"

    def call_llm(
        self,
        model: str = None,
        reasoning_effort: str = "low",
        system_instructions: str = None,
        user_prompt: str = None,
        tools: list = None,
        json_schema: dict = None,
        schema_name: str = None,
        retries: int = 2,
        user=None,
    ):
        if not json_schema or not schema_name:
            raise ValueError("json_schema and schema_name are required for structured calls")
        if not user:
            raise ValueError("user parameter is required for usage tracking")

        model = model or self.model
        provider = get_provider(model)
        result = provider.call_llm(
            model=model,
            reasoning_effort=reasoning_effort,
            system_instructions=system_instructions,
            user_prompt=user_prompt,
            tools=tools,
            json_schema=json_schema,
            schema_name=schema_name,
            retries=retries,
        )
        call_log = _result_to_call_log(result, _get_caller_info())
        if result.get("succeeded"):
            _track_usage(user, result.get("llm_cost_usd", 0), result.get("total_tokens", 0))
        return call_log

    def call_llm_stream(
        self,
        model: str | None = None,
        reasoning_effort: str = "low",
        system_instructions: str | None = None,
        user_prompt: str | None = None,
        tools: list | None = None,
        json_schema: dict | None = None,
        schema_name: str | None = None,
        retries: int = 2,
        user=None,
    ):
        if not json_schema or not schema_name:
            raise ValueError("json_schema and schema_name are required for streaming calls")
        if not user:
            raise ValueError("user parameter is required for usage tracking")

        model = model or self.model
        provider = get_provider(model)
        for event_type, event in provider.call_llm_stream(
            model=model,
            reasoning_effort=reasoning_effort,
            system_instructions=system_instructions,
            user_prompt=user_prompt,
            tools=tools,
            json_schema=json_schema,
            schema_name=schema_name,
            retries=retries,
        ):
            if event_type == "final":
                result = event
                call_log = _result_to_call_log(result, _get_caller_info())
                if result.get("succeeded"):
                    _track_usage(user, result.get("llm_cost_usd", 0), result.get("total_tokens", 0))
                yield ("final", {"response": result.get("response"), "call_log": call_log})
            else:
                yield (event_type, event)
