"""
OpenAI provider: Responses API, structured output, tool calling, streaming.
Tool execution uses the shared tool_registry; tool schema and invocation are OpenAI-specific.
"""

import json
import logging
import random
import time

from django.conf import settings

from llm_service.registry import get_price_plan
from llm_service.tools.registry import tool_registry
from llm_service.utils.llm_pricing import LLMPricing

logger = logging.getLogger(__name__)


class OpenAIProvider:
    """
    OpenAI Responses API provider. Handles API calls, retries, tool execution
    (via shared tool_registry), and returns neutral result dicts for the router
    to turn into LLMCallLog and usage tracking.
    """

    def __init__(self):
        try:
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("httpx").setLevel(logging.ERROR)
            logging.getLogger("httpcore").setLevel(logging.ERROR)
            import openai
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise Exception("OPENAI_API_KEY not found in environment variables")
            self.client = openai.OpenAI(api_key=api_key, timeout=60.0)
            self.default_model = "gpt-5"
        except Exception as e:
            raise Exception(f"Failed to initialize OpenAI provider: {e}") from e

    def call_llm(
        self,
        model: str | None = None,
        reasoning_effort: str = "low",
        system_instructions: str | None = None,
        user_prompt: str | None = None,
        tools: list | None = None,
        json_schema: dict | None = None,
        schema_name: str | None = None,
        retries: int = 2,
    ):
        """
        Make a structured OpenAI call. Returns a result dict for the router to
        create LLMCallLog and track usage. Does not touch Django models.
        """
        model = model or self.default_model
        start_time = time.monotonic()
        text_format = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            }
        }
        api_params = {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "instructions": system_instructions,
            "input": user_prompt,
            "text": text_format,
        }
        if tools is not None:
            api_params["tools"] = tools

        last_exception = None
        rate_limit_attempts = 0
        for attempt in range(retries + 1):
            try:
                response = self._make_api_call_with_tools(api_params)
                parsed_json = None
                if response.status == "completed" and response.output:
                    for output_item in response.output:
                        if hasattr(output_item, "content") and output_item.content:
                            for content_item in output_item.content:
                                if hasattr(content_item, "text") and content_item.text:
                                    parsed_json = json.loads(content_item.text)
                                    break
                            if parsed_json:
                                break
                token_usage = self._extract_token_usage(response)
                cost = self._calculate_cost(model, token_usage)
                duration = time.monotonic() - start_time
                logger.info(
                    "LLM called: model=%s, tokens=%d, cost=$%.6f, status=%s, time=%.2fs",
                    model,
                    token_usage.get("total_tokens", 0),
                    cost,
                    "success" if response.status == "completed" else "failed",
                    duration,
                )
                return {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "system_instructions": system_instructions,
                    "user_prompt": user_prompt,
                    "json_schema": json_schema,
                    "schema_name": schema_name,
                    "parsed_json": parsed_json,
                    "input_tokens": token_usage.get("input_tokens", 0),
                    "output_tokens": token_usage.get("output_tokens", 0),
                    "cached_tokens": token_usage.get("cached_tokens", 0),
                    "reasoning_tokens": token_usage.get("reasoning_tokens", 0),
                    "total_tokens": token_usage.get("total_tokens", 0),
                    "succeeded": response.status == "completed",
                    "llm_cost_usd": cost,
                    "response_time_seconds": duration,
                    "raw_response": self._response_to_dict(response),
                }
            except Exception as e:
                last_exception = e
                if self._is_rate_limit_error(e):
                    rate_limit_attempts += 1
                    if attempt < retries:
                        logger.info("Rate limit hit on attempt %s: %s", attempt + 1, e)
                        self._handle_rate_limit_retry(rate_limit_attempts - 1)
                        continue
                elif self._is_timeout_error(e):
                    if attempt < retries:
                        delay = min(5 + (attempt * 2), 15)
                        time.sleep(delay)
                        continue
                else:
                    if attempt < retries:
                        time.sleep(1)
                        continue
                duration = time.monotonic() - start_time
                return {
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "system_instructions": system_instructions,
                    "user_prompt": user_prompt,
                    "json_schema": json_schema,
                    "schema_name": schema_name,
                    "parsed_json": None,
                    "input_tokens": 0,
                    "output_tokens": 0,
                    "cached_tokens": 0,
                    "reasoning_tokens": 0,
                    "total_tokens": 0,
                    "succeeded": False,
                    "llm_cost_usd": 0,
                    "response_time_seconds": duration,
                    "raw_response": {"error": str(e), "attempts": attempt + 1, "rate_limit_attempts": rate_limit_attempts},
                }
        raise Exception(f"LLM call failed after {retries + 1} attempts. Last error: {last_exception}")  # noqa: B904

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
    ):
        """
        Stream a structured OpenAI call. Yields (event_type, event); ends with
        ("final", result_dict) for the router to create LLMCallLog and track usage.
        """
        model = model or self.default_model
        start_time = time.monotonic()
        text_format = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            }
        }
        conversation_input = [
            {"role": "system", "content": system_instructions or ""},
            {"role": "user", "content": user_prompt or ""},
        ]
        api_params = {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "input": conversation_input,
            "text": text_format,
        }
        if tools is not None:
            api_params["tools"] = tools

        last_response = None
        stream_failure_exception = None
        try:
            while True:
                function_calls = {}
                completed_tool_calls = []
                round_response = None
                round_succeeded = False
                last_round_exception = None
                rate_limit_attempts = 0

                for attempt in range(retries + 1):
                    try:
                        events_yielded_this_attempt = False
                        with self.client.responses.stream(**api_params) as stream:
                            for event in stream:
                                event_type = event.type
                                yield (event_type, event)
                                events_yielded_this_attempt = True

                                if event_type == "response.output_item.added":
                                    item = getattr(event, "item", None)
                                    if item and getattr(item, "type", None) == "function_call":
                                        output_index = getattr(event, "output_index", None)
                                        if output_index is not None:
                                            function_calls[output_index] = {
                                                "id": getattr(item, "id", None),
                                                "call_id": getattr(item, "call_id", None),
                                                "name": getattr(item, "name", None),
                                                "arguments": getattr(item, "arguments", "") or "",
                                            }
                                elif event_type == "response.function_call_arguments.delta":
                                    output_index = getattr(event, "output_index", None)
                                    delta = getattr(event, "delta", "")
                                    if output_index is not None and output_index in function_calls:
                                        function_calls[output_index]["arguments"] += delta
                                elif event_type == "response.function_call_arguments.done":
                                    output_index = getattr(event, "output_index", None)
                                    arguments = getattr(event, "arguments", "")
                                    if output_index is not None and output_index in function_calls:
                                        function_calls[output_index]["arguments"] = arguments
                                        completed_tool_calls.append(function_calls[output_index])

                            round_response = stream.get_final_response()
                            round_succeeded = True
                            break

                    except Exception as e:
                        last_round_exception = e
                        if events_yielded_this_attempt:
                            yield ("response.error", {"error": str(e), "retry_aborted": True})
                            break
                        if self._is_rate_limit_error(e):
                            rate_limit_attempts += 1
                            if attempt < retries:
                                self._handle_rate_limit_retry(rate_limit_attempts - 1)
                                continue
                        elif self._is_timeout_error(e):
                            if attempt < retries:
                                delay = min(5 + (attempt * 2), 15)
                                time.sleep(delay)
                                continue
                        else:
                            if self._is_retryable_error(e) and attempt < retries:
                                time.sleep(1)
                                continue
                            break

                if not round_succeeded:
                    stream_failure_exception = last_round_exception
                    break

                last_response = round_response
                if not completed_tool_calls and last_response:
                    for output_item in last_response.output:
                        if getattr(output_item, "type", None) == "function_call":
                            completed_tool_calls.append({
                                "id": getattr(output_item, "id", None),
                                "call_id": getattr(output_item, "call_id", None),
                                "name": getattr(output_item, "name", None),
                                "arguments": getattr(output_item, "arguments", ""),
                            })

                if completed_tool_calls:
                    conversation_input = list(conversation_input)
                    conversation_input.extend(last_response.output)
                    for tool_call_data in completed_tool_calls:
                        tool_result = tool_registry.execute(
                            tool_call_data["name"],
                            tool_call_data["arguments"],
                        )
                        conversation_input.append({
                            "type": "function_call_output",
                            "call_id": tool_call_data["call_id"],
                            "output": tool_result if tool_result is not None else "Tool execution failed",
                        })
                    api_params["input"] = conversation_input
                    function_calls = {}
                    completed_tool_calls = []
                    continue
                break

            parsed_json = None
            if last_response and last_response.status == "completed" and last_response.output:
                for output_item in last_response.output:
                    if hasattr(output_item, "content") and output_item.content:
                        for content_item in output_item.content:
                            if hasattr(content_item, "text") and content_item.text:
                                try:
                                    parsed_json = json.loads(content_item.text)
                                except json.JSONDecodeError:
                                    pass
                                break
                        if parsed_json is not None:
                            break

            token_usage = self._extract_token_usage(last_response) if last_response else {}
            cost = self._calculate_cost(model, token_usage)
            duration = time.monotonic() - start_time

            if not last_response or stream_failure_exception:
                raw_response = {
                    "error": str(stream_failure_exception) if stream_failure_exception else "Stream failed",
                    "retries_exhausted": True,
                    "attempts": retries + 1,
                }
                succeeded = False
            else:
                raw_response = self._response_to_dict(last_response)
                succeeded = bool(last_response.status == "completed")

            result = {
                "model": model,
                "reasoning_effort": reasoning_effort,
                "system_instructions": system_instructions,
                "user_prompt": user_prompt,
                "json_schema": json_schema,
                "schema_name": schema_name,
                "parsed_json": parsed_json,
                "input_tokens": token_usage.get("input_tokens", 0),
                "output_tokens": token_usage.get("output_tokens", 0),
                "cached_tokens": token_usage.get("cached_tokens", 0),
                "reasoning_tokens": token_usage.get("reasoning_tokens", 0),
                "total_tokens": token_usage.get("total_tokens", 0),
                "succeeded": succeeded,
                "llm_cost_usd": cost,
                "response_time_seconds": duration,
                "raw_response": raw_response,
                "response": last_response,
            }
            logger.info(
                "LLM stream completed: model=%s, tokens=%d, cost=$%.6f, status=%s, time=%.2fs",
                model,
                token_usage.get("total_tokens", 0),
                cost,
                "success" if succeeded else "failed",
                duration,
            )
            yield ("final", result)

        except Exception as e:
            duration = time.monotonic() - start_time
            logger.info("LLM stream failed: model=%s, time=%.2fs, error=%s", model, duration, e)
            yield ("final", {
                "model": model,
                "reasoning_effort": reasoning_effort,
                "system_instructions": system_instructions,
                "user_prompt": user_prompt,
                "json_schema": json_schema,
                "schema_name": schema_name,
                "parsed_json": None,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_tokens": 0,
                "reasoning_tokens": 0,
                "total_tokens": 0,
                "succeeded": False,
                "llm_cost_usd": 0,
                "response_time_seconds": duration,
                "raw_response": {"error": str(e)},
                "response": None,
            })

    def _make_api_call_with_tools(self, api_params):
        if isinstance(api_params["input"], str):
            conversation_history = [{"role": "user", "content": api_params["input"]}]
            api_params = {**api_params, "input": conversation_history}
        else:
            conversation_history = list(api_params["input"]) if isinstance(api_params["input"], list) else [{"role": "user", "content": str(api_params["input"])}]

        response = self.client.responses.parse(**api_params)
        while True:
            tool_calls = [o for o in response.output if getattr(o, "type", None) == "function_call"]
            if not tool_calls:
                return response
            conversation_history.extend(response.output)
            for tool_call in tool_calls:
                tool_result = tool_registry.execute(tool_call.name, tool_call.arguments)
                conversation_history.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": tool_result if tool_result is not None else "Tool execution failed",
                })
            follow_up_params = {
                "model": api_params["model"],
                "reasoning": api_params["reasoning"],
                "instructions": api_params["instructions"],
                "input": conversation_history,
                "text": api_params["text"],
            }
            if "tools" in api_params:
                follow_up_params["tools"] = api_params["tools"]
            response = self.client.responses.parse(**follow_up_params)

    def _handle_rate_limit_retry(self, attempt, base_delay=1, max_delay=60):
        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = random.uniform(0, delay * 0.25)
        time.sleep(delay + jitter)

    def _is_rate_limit_error(self, exception):
        s = str(exception).lower()
        return "429" in s or "rate limit" in s or "too many requests" in s or "quota exceeded" in s

    def _is_timeout_error(self, exception):
        s = str(exception).lower()
        t = type(exception).__name__.lower()
        return "timeout" in s or "timed out" in s or "timeout" in t or "read timeout" in s or "connection timeout" in s

    def _is_retryable_error(self, exception):
        s = str(exception).lower()
        t = type(exception).__name__.lower()
        if any(x in s for x in ["connection", "network", "dns", "resolve", "unreachable"]):
            return True
        if "503" in s or "service unavailable" in s:
            return True
        if "apiconnectionerror" in t:
            return True
        if "500" in s or "internal server error" in s:
            return True
        return False

    def _extract_token_usage(self, response):
        token_usage = {"input_tokens": 0, "output_tokens": 0, "cached_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0}
        if getattr(response, "usage", None):
            u = response.usage
            token_usage["input_tokens"] = getattr(u, "input_tokens", 0)
            token_usage["output_tokens"] = getattr(u, "output_tokens", 0)
            token_usage["total_tokens"] = getattr(u, "total_tokens", 0)
            if getattr(u, "input_tokens_details", None):
                token_usage["cached_tokens"] = getattr(u.input_tokens_details, "cached_tokens", 0)
            if getattr(u, "output_tokens_details", None):
                token_usage["reasoning_tokens"] = getattr(u.output_tokens_details, "reasoning_tokens", 0)
        return token_usage

    def _calculate_cost(self, model, token_usage):
        plan = get_price_plan(model) or model
        return LLMPricing.calculate_cost(
            model_name=plan,
            input_tokens=token_usage.get("input_tokens", 0),
            cached_input_tokens=token_usage.get("cached_tokens", 0),
            output_tokens=token_usage.get("output_tokens", 0),
            reasoning_tokens=token_usage.get("reasoning_tokens", 0),
        )

    def _response_to_dict(self, response):
        try:
            return {
                "id": response.id,
                "created_at": response.created_at,
                "model": response.model,
                "status": response.status,
                "usage": {
                    "input_tokens": response.usage.input_tokens if response.usage else 0,
                    "output_tokens": response.usage.output_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                } if response.usage else None,
            }
        except Exception:
            return {"error": "Failed to serialize response"}
