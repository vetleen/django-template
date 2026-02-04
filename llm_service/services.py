"""
LLM Service for Django template project

This service handles all LLM interactions using OpenAI's Responses API.
Provides both blocking (call_llm) and streaming (call_llm_stream) interfaces
with structured JSON output, tool calling, retry logic, and usage tracking.
"""

import inspect
import json
import time
import random
import logging
from django.conf import settings
from llm_service.models import LLMCallLog
from llm_service.utils.llm_pricing import LLMPricing
from llm_service.tools.registry import tool_registry

logger = logging.getLogger(__name__)

class LLMService:
    """
    Central service for all LLM operations.
    
    Provides a clean API for interacting with OpenAI's Responses API:
    - Structured JSON output with schema validation
    - Tool calling with automatic execution
    - Streaming support with real-time event handling
    - Retry logic with exponential backoff
    - Usage tracking and cost calculation
    
    Example:
        service = LLMService()
        call_log = service.call_llm(
            user_prompt="Hello",
            json_schema={"type": "object", "properties": {"message": {"type": "string"}}},
            schema_name="greeting",
            user=request.user,
        )
    """
    
    def __init__(self):
        """Initialize the LLM service."""
        try:
            # Disable OpenAI HTTP request logging
            logging.getLogger("openai").setLevel(logging.ERROR)
            logging.getLogger("httpx").setLevel(logging.ERROR)
            logging.getLogger("httpcore").setLevel(logging.ERROR)
            
            # Import openai here to avoid hanging during module import
            import openai
            
            # Check if API key is provided
            api_key = settings.OPENAI_API_KEY
            if not api_key:
                raise Exception("OPENAI_API_KEY not found in environment variables")
            
            # Configure OpenAI client
            self.client = openai.OpenAI(
                api_key=api_key,
                timeout=60.0
            )
            self.model = "gpt-5"
            
        except Exception as e:
            raise Exception(f"Failed to initialize LLM service: {e}")
    
    def call_llm(self, model: str = None, reasoning_effort: str = "low",
                 system_instructions: str = None, user_prompt: str = None, tools: list = None,
                 json_schema: dict = None, schema_name: str = None, retries: int = 2, user=None):
        """
        Make a structured LLM call that returns parsed JSON data using responses.parse().
        
        Supports tool calling with automatic tool execution and follow-up calls.
        Includes retry logic with exponential backoff for rate limits and incremental delays
        for timeouts. Returns a failed LLMCallLog if all retries are exhausted instead of
        raising an exception.
        
        Args:
            model: The model to use (default: gpt-5)
            reasoning_effort: The reasoning effort level (default: low)
            system_instructions: System instructions that set the LLM's context and behavior
            user_prompt: The user's input/prompt for the LLM to process
            tools: Optional list of tool definitions for the LLM to use
            json_schema: JSON schema for structured output (required)
            schema_name: Name for the schema (required)
            retries: Number of retry attempts before failing (default: 2)
            user: User instance for usage tracking (required)
            
        Returns:
            LLMCallLog object with parsed_json, token usage, cost, and success status
            
        Raises:
            ValueError: If json_schema, schema_name, or user is missing
        """
        if not json_schema or not schema_name:
            raise ValueError("json_schema and schema_name are required for structured calls")
        
        if not user:
            raise ValueError("user parameter is required for usage tracking")
        
        # Use self.model as default if no model specified
        model = model or self.model
        
        # Track start time for logging
        start_time = time.monotonic()
        
        # Get caller information
        caller_info = self._get_caller_info()
        
        # Prepare text format for structured output
        text_format = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": json_schema,
                "strict": True
            }
        }
        
        # Prepare API call parameters
        api_params = {
            "model": model,
            "reasoning": {"effort": reasoning_effort},
            "instructions": system_instructions,
            "input": user_prompt,
            "text": text_format
        }
        
        # Only add tools if provided
        if tools is not None:
            api_params["tools"] = tools
        
        # Retry logic with exponential backoff for rate limits
        last_exception = None
        rate_limit_attempts = 0
        
        for attempt in range(retries + 1):  # +1 for initial attempt
            try:
                # Make initial API call
                response = self._make_api_call_with_tools(api_params)
                
                # Extract parsed JSON from the final response
                parsed_json = None
                if response.status == "completed" and response.output:
                    # Get the JSON text from the first output message
                    for output_item in response.output:
                        if hasattr(output_item, 'content') and output_item.content:
                            for content_item in output_item.content:
                                if hasattr(content_item, 'text') and content_item.text:
                                    parsed_json = json.loads(content_item.text)
                                    break
                            if parsed_json:
                                break
                
                # Extract token usage
                token_usage = self._extract_token_usage(response)
                
                # Calculate cost
                cost = self._calculate_cost(model, token_usage)
                
                # Calculate duration
                duration = time.monotonic() - start_time
                
                # Create and save LLMCallLog
                call_log = LLMCallLog.objects.create(
                    raw_response=self._response_to_dict(response),
                    caller=caller_info,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    system_instructions=system_instructions,
                    user_prompt=user_prompt,
                    json_schema=json_schema,
                    schema_name=schema_name,
                    parsed_json=parsed_json,
                    input_tokens=token_usage.get('input_tokens', 0),
                    output_tokens=token_usage.get('output_tokens', 0),
                    cached_tokens=token_usage.get('cached_tokens', 0),
                    reasoning_tokens=token_usage.get('reasoning_tokens', 0),
                    total_tokens=token_usage.get('total_tokens', 0),
                    succeeded=response.status == "completed",
                    llm_cost_usd=cost,
                    response_time_seconds=duration,
                )
                
                # Log the call
                status = "success" if response.status == "completed" else "failed"
                logger.info(
                    "LLM called: model=%s, tokens=%d, cost=$%.6f, status=%s, time=%.2fs",
                    model,
                    token_usage.get('total_tokens', 0),
                    cost,
                    status,
                    duration,
                )
                
                # Track usage per user
                if response.status == "completed":
                    from llm_service.models import UserMonthlyUsage
                    from django.db import transaction
                    from decimal import Decimal
                    from datetime import datetime
                    
                    now = datetime.now()
                    year = now.year
                    month = now.month
                    
                    with transaction.atomic():
                        try:
                            # Try to get existing usage record
                            usage = UserMonthlyUsage.objects.select_for_update().get(
                                user=user,
                                year=year,
                                month=month
                            )
                            # Update existing record
                            usage.total_cost_usd += Decimal(str(cost))
                            usage.total_tokens += token_usage.get('total_tokens', 0)
                            usage.total_calls += 1
                            usage.save()
                        except UserMonthlyUsage.DoesNotExist:
                            # Create new record
                            UserMonthlyUsage.objects.create(
                                user=user,
                                year=year,
                                month=month,
                                total_cost_usd=Decimal(str(cost)),
                                total_tokens=token_usage.get('total_tokens', 0),
                                total_calls=1
                            )
                
                return call_log
                
            except Exception as e:
                last_exception = e
                
                # Check if this is a rate limit error
                if self._is_rate_limit_error(e):
                    rate_limit_attempts += 1
                    if attempt < retries:
                        print(f"Rate limit hit on attempt {attempt + 1}: {e}")
                        self._handle_rate_limit_retry(rate_limit_attempts - 1)
                        continue
                # Check if this is a timeout error
                elif self._is_timeout_error(e):
                    if attempt < retries:
                        print(f"LLM call attempt {attempt + 1} failed: Request timed out. Retrying...")
                        # Add backoff delay for timeout retries
                        timeout_attempt = attempt
                        delay = min(5 + (timeout_attempt * 2), 15)  # 5s, 7s, 9s, max 15s
                        print(f"Waiting {delay}s before retry...")
                        time.sleep(delay)
                        continue
                else:
                    # Regular error handling (no backoff for other errors)
                    if attempt < retries:
                        print(f"LLM call attempt {attempt + 1} failed: {e}. Retrying...")
                    else:
                        # Final attempt failed, create a failed log entry
                        duration = time.monotonic() - start_time
                        try:
                            failed_log = LLMCallLog.objects.create(
                                raw_response={'error': str(e), 'attempts': attempt + 1, 'rate_limit_attempts': rate_limit_attempts},
                                caller=caller_info,
                                model=model,
                                reasoning_effort=reasoning_effort,
                                system_instructions=system_instructions,
                                user_prompt=user_prompt,
                                json_schema=json_schema,
                                schema_name=schema_name,
                                parsed_json=None,
                                input_tokens=0,
                                output_tokens=0,
                                cached_tokens=0,
                                reasoning_tokens=0,
                                total_tokens=0,
                                succeeded=False,
                                llm_cost_usd=0,
                                response_time_seconds=duration,
                            )
                            logger.info(
                                "LLM called: model=%s, tokens=0, cost=$0.000000, status=failed, time=%.2fs",
                                model,
                                duration,
                            )
                            return failed_log
                        except Exception as db_error:
                            # If we can't even save the failed log, raise the original error
                            raise Exception(f"LLM call failed after {retries + 1} attempts. Last error: {e}. Additionally, failed to save error log: {db_error}")
        
        # This should never be reached, but just in case
        raise Exception(f"LLM call failed after {retries + 1} attempts. Last error: {last_exception}")

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
        """
        Stream a structured LLM call, yielding events as they arrive.

        Supports immediate tool calling: detects function calls as they stream,
        accumulates arguments via delta events, executes tools when arguments are
        complete, and immediately starts follow-up streams until no more tool calls.

        Retry logic: Each stream round is retried independently if it fails before
        any events are yielded. Once events start streaming, retries are disabled
        for that round to avoid duplicate events. Callers can also handle retries
        manually by checking the final call_log and calling again if needed.

        Yields (event_type, event) for each stream event. Event types follow the
        Responses API (e.g. "response.output_text.delta", "response.refusal.delta",
        "response.output_item.added", "response.function_call_arguments.delta",
        "response.function_call_arguments.done", "response.error", "response.completed").
        When done, yields ("final", {"response": response, "call_log": call_log}).

        Args:
            model: Model to use (default: self.model).
            reasoning_effort: Reasoning effort level.
            system_instructions: System instructions.
            user_prompt: User prompt.
            tools: Optional list of tool definitions (same as call_llm).
            json_schema: JSON schema for structured output (required).
            schema_name: Schema name (required).
            retries: Number of retry attempts per stream round (default: 2).
            user: User instance for usage tracking (required).

        Yields:
            (event_type, event) for stream events; ("final", {"response", "call_log"}) at end.
        """
        if not json_schema or not schema_name:
            raise ValueError("json_schema and schema_name are required for streaming calls")
        if not user:
            raise ValueError("user parameter is required for usage tracking")

        model = model or self.model
        start_time = time.monotonic()
        caller_info = self._get_caller_info()

        text_format = {
            "format": {
                "type": "json_schema",
                "name": schema_name,
                "schema": json_schema,
                "strict": True,
            }
        }

        # Build input as message list (stream API uses input=list of messages)
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
                # Stream this round with retry logic per round
                function_calls = {}  # Maps output_index -> function call data
                completed_tool_calls = []  # Tool calls ready to execute
                round_response = None
                round_succeeded = False
                last_round_exception = None
                rate_limit_attempts = 0
                
                # Retry loop for this stream round
                for attempt in range(retries + 1):  # +1 for initial attempt
                    try:
                        events_yielded_this_attempt = False
                        
                        with self.client.responses.stream(**api_params) as stream:
                            for event in stream:
                                event_type = event.type
                                yield (event_type, event)
                                events_yielded_this_attempt = True
                                
                                # Handle function call events for immediate execution
                                if event_type == "response.output_item.added":
                                    # Check if this is a function call
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
                                    # Accumulate argument deltas
                                    output_index = getattr(event, "output_index", None)
                                    delta = getattr(event, "delta", "")
                                    if output_index is not None and output_index in function_calls:
                                        function_calls[output_index]["arguments"] += delta
                                
                                elif event_type == "response.function_call_arguments.done":
                                    # Function call arguments are complete - ready to execute
                                    output_index = getattr(event, "output_index", None)
                                    arguments = getattr(event, "arguments", "")
                                    if output_index is not None and output_index in function_calls:
                                        function_calls[output_index]["arguments"] = arguments
                                        completed_tool_calls.append(function_calls[output_index])
                            
                            round_response = stream.get_final_response()
                            round_succeeded = True
                            break  # Success, exit retry loop
                            
                    except Exception as e:
                        last_round_exception = e
                        
                        # If events were already yielded, don't retry (would cause duplicates)
                        if events_yielded_this_attempt:
                            # Yield error event and break retry loop
                            yield ("response.error", {"error": str(e), "retry_aborted": True})
                            break
                        
                        # Check if this is a retryable error
                        if self._is_rate_limit_error(e):
                            rate_limit_attempts += 1
                            if attempt < retries:
                                logger.info(f"Stream round rate limit hit on attempt {attempt + 1}: {e}")
                                self._handle_rate_limit_retry(rate_limit_attempts - 1)
                                continue
                        elif self._is_timeout_error(e):
                            if attempt < retries:
                                logger.info(f"Stream round timeout on attempt {attempt + 1}: {e}")
                                timeout_attempt = attempt
                                delay = min(5 + (timeout_attempt * 2), 15)  # 5s, 7s, 9s, max 15s
                                time.sleep(delay)
                                continue
                        else:
                            # Check if error is retryable (network errors, service unavailable)
                            if self._is_retryable_error(e) and attempt < retries:
                                logger.info(f"Stream round error on attempt {attempt + 1}: {e}. Retrying...")
                                # Small delay for non-rate-limit retries
                                time.sleep(1)
                                continue
                            else:
                                # Non-retryable error or retries exhausted
                                if attempt < retries:
                                    logger.info(f"Stream round non-retryable error on attempt {attempt + 1}: {e}")
                                break
                
                # Check if round succeeded
                if not round_succeeded:
                    # All retries exhausted for this round
                    stream_failure_exception = last_round_exception
                    logger.warning(f"Stream round failed after {retries + 1} attempts: {last_round_exception}")
                    # Break outer loop - will create failed log
                    break
                
                last_response = round_response
                
                # Also check final response for any tool calls we might have missed
                if not completed_tool_calls and last_response:
                    for output_item in last_response.output:
                        if getattr(output_item, "type", None) == "function_call":
                            completed_tool_calls.append({
                                "id": getattr(output_item, "id", None),
                                "call_id": getattr(output_item, "call_id", None),
                                "name": getattr(output_item, "name", None),
                                "arguments": getattr(output_item, "arguments", ""),
                            })
                
                # Execute completed tool calls immediately
                if completed_tool_calls:
                    # Extend conversation with model output from last_response
                    conversation_input = list(conversation_input)
                    conversation_input.extend(last_response.output)
                    
                    # Execute tools and add results
                    for tool_call_data in completed_tool_calls:
                        tool_result = tool_registry.execute(
                            tool_call_data["name"],
                            tool_call_data["arguments"]
                        )
                        conversation_input.append({
                            "type": "function_call_output",
                            "call_id": tool_call_data["call_id"],
                            "output": tool_result if tool_result is not None else "Tool execution failed",
                        })
                    
                    # Start follow-up stream immediately
                    api_params["input"] = conversation_input
                    # Reset tracking for next round
                    function_calls = {}
                    completed_tool_calls = []
                    continue
                
                # No tool calls, we're done
                break

            # Build LLMCallLog and usage from last_response
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

            # Create call log - failed if no response or retries exhausted
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
            
            call_log = LLMCallLog.objects.create(
                raw_response=raw_response,
                caller=caller_info,
                model=model,
                reasoning_effort=reasoning_effort,
                system_instructions=system_instructions,
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name=schema_name,
                parsed_json=parsed_json,
                input_tokens=token_usage.get("input_tokens", 0),
                output_tokens=token_usage.get("output_tokens", 0),
                cached_tokens=token_usage.get("cached_tokens", 0),
                reasoning_tokens=token_usage.get("reasoning_tokens", 0),
                total_tokens=token_usage.get("total_tokens", 0),
                succeeded=succeeded,
                llm_cost_usd=cost,
                response_time_seconds=duration,
            )

            logger.info(
                "LLM stream completed: model=%s, tokens=%d, cost=$%.6f, status=%s, time=%.2fs",
                model,
                token_usage.get("total_tokens", 0),
                cost,
                "success" if (last_response and last_response.status == "completed") else "failed",
                duration,
            )

            if last_response and last_response.status == "completed":
                from django.db import transaction
                from decimal import Decimal
                from datetime import datetime

                from llm_service.models import UserMonthlyUsage

                now = datetime.now()
                with transaction.atomic():
                    try:
                        usage = UserMonthlyUsage.objects.select_for_update().get(
                            user=user, year=now.year, month=now.month
                        )
                        usage.total_cost_usd += Decimal(str(cost))
                        usage.total_tokens += token_usage.get("total_tokens", 0)
                        usage.total_calls += 1
                        usage.save()
                    except UserMonthlyUsage.DoesNotExist:
                        UserMonthlyUsage.objects.create(
                            user=user,
                            year=now.year,
                            month=now.month,
                            total_cost_usd=Decimal(str(cost)),
                            total_tokens=token_usage.get("total_tokens", 0),
                            total_calls=1,
                        )

            yield ("final", {"response": last_response, "call_log": call_log})

        except Exception as e:
            from decimal import Decimal

            duration = time.monotonic() - start_time
            logger.info(
                "LLM stream failed: model=%s, time=%.2fs, error=%s",
                model,
                duration,
                e,
            )
            try:
                failed_log = LLMCallLog.objects.create(
                    raw_response={"error": str(e)},
                    caller=caller_info,
                    model=model,
                    reasoning_effort=reasoning_effort,
                    system_instructions=system_instructions,
                    user_prompt=user_prompt,
                    json_schema=json_schema,
                    schema_name=schema_name,
                    parsed_json=None,
                    input_tokens=0,
                    output_tokens=0,
                    cached_tokens=0,
                    reasoning_tokens=0,
                    total_tokens=0,
                    succeeded=False,
                    llm_cost_usd=Decimal("0"),
                    response_time_seconds=duration,
                )
                yield ("final", {"response": None, "call_log": failed_log})
            except Exception as db_err:
                raise Exception(f"Stream failed: {e}. Failed to save error log: {db_err}") from e
    
    def _make_api_call_with_tools(self, api_params):
        """
        Make API call and handle tool calling if needed.
        
        Args:
            api_params: Parameters for the API call
            
        Returns:
            Final response after handling all tool calls
        """
        # Convert input to list format if it's a string
        if isinstance(api_params["input"], str):
            conversation_history = [{"role": "user", "content": api_params["input"]}]
            # Update api_params to use list format for consistency
            api_params = api_params.copy()
            api_params["input"] = conversation_history
        else:
            conversation_history = api_params["input"].copy() if isinstance(api_params["input"], list) else [{"role": "user", "content": str(api_params["input"])}]
        
        # Make initial API call
        response = self.client.responses.parse(**api_params)
        
        # Handle tool calls in a loop until no more tool calls
        while True:
            # Check if response contains tool calls
            tool_calls = []
            for output_item in response.output:
                if hasattr(output_item, 'type') and output_item.type == "function_call":
                    tool_calls.append(output_item)
            
            if not tool_calls:
                # No more tool calls, return the final response
                return response
            
            # Add response output to conversation history
            conversation_history.extend(response.output)
            
            # Execute tool calls and add results
            for tool_call in tool_calls:
                tool_result = tool_registry.execute(tool_call.name, tool_call.arguments)
                
                # Add tool call output to conversation history
                conversation_history.append({
                    "type": "function_call_output",
                    "call_id": tool_call.call_id,
                    "output": tool_result if tool_result is not None else "Tool execution failed"
                })
            
            # Make follow-up API call with tool results
            follow_up_params = {
                "model": api_params["model"],
                "reasoning": api_params["reasoning"],
                "instructions": api_params["instructions"],
                "input": conversation_history,
                "text": api_params["text"]
            }
            
            # Add tools if they were in the original call
            if "tools" in api_params:
                follow_up_params["tools"] = api_params["tools"]
            
            response = self.client.responses.parse(**follow_up_params)
    
    def _handle_rate_limit_retry(self, attempt, base_delay=1, max_delay=60):
        """
        Handle exponential backoff for rate limiting.
        
        Args:
            attempt: Current attempt number (0-based)
            base_delay: Base delay in seconds (default: 1)
            max_delay: Maximum delay in seconds (default: 60)
        """
        delay = min(base_delay * (2 ** attempt), max_delay)
        jitter = random.uniform(0, delay * 0.25)
        total_delay = delay + jitter
        print(f"Rate limited, waiting {total_delay:.1f}s before retry...")
        time.sleep(total_delay)
    
    def _is_rate_limit_error(self, exception):
        """
        Check if an exception is a rate limit error.
        
        Args:
            exception: The exception to check
            
        Returns:
            True if it's a rate limit error, False otherwise
        """
        error_str = str(exception).lower()
        return (
            "429" in error_str or
            "rate limit" in error_str or
            "too many requests" in error_str or
            "quota exceeded" in error_str
        )
    
    def _is_timeout_error(self, exception):
        """
        Check if an exception is a timeout error.
        
        Args:
            exception: The exception to check
            
        Returns:
            True if it's a timeout error, False otherwise
        """
        error_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        return (
            "timeout" in error_str or
            "timed out" in error_str or
            "timeout" in exception_type or
            "read timeout" in error_str or
            "connection timeout" in error_str
        )
    
    def _is_retryable_error(self, exception):
        """
        Check if an exception is retryable (network errors, service unavailable).
        Does not include rate limits or timeouts (handled separately).
        
        Args:
            exception: The exception to check
            
        Returns:
            True if retryable, False otherwise
        """
        error_str = str(exception).lower()
        exception_type = type(exception).__name__.lower()
        
        # Network/connection errors
        if any(term in error_str for term in ["connection", "network", "dns", "resolve", "unreachable"]):
            return True
        
        # Service unavailable
        if "503" in error_str or "service unavailable" in error_str:
            return True
        
        # APIConnectionError from OpenAI SDK
        if "apiconnectionerror" in exception_type:
            return True
        
        # Internal server errors (sometimes retryable)
        if "500" in error_str or "internal server error" in error_str:
            return True
        
        return False
    
    def _get_caller_info(self):
        """Get caller information using inspect.stack()"""
        try:
            frame = inspect.stack()[2]  # Skip this method and call_llm
            filename = frame.filename
            function_name = frame.function
            line_number = frame.lineno
            return f"{filename}:{function_name}:{line_number}"
        except (IndexError, AttributeError):
            return "unknown"
    
    def _extract_token_usage(self, response):
        """Extract token usage information from response"""
        token_usage = {
            'input_tokens': 0,
            'output_tokens': 0,
            'cached_tokens': 0,
            'reasoning_tokens': 0,
            'total_tokens': 0
        }
        
        if hasattr(response, 'usage') and response.usage:
            usage = response.usage
            token_usage['input_tokens'] = getattr(usage, 'input_tokens', 0)
            token_usage['output_tokens'] = getattr(usage, 'output_tokens', 0)
            token_usage['total_tokens'] = getattr(usage, 'total_tokens', 0)
            
            # Extract cached tokens from input_tokens_details
            if hasattr(usage, 'input_tokens_details') and usage.input_tokens_details:
                token_usage['cached_tokens'] = getattr(usage.input_tokens_details, 'cached_tokens', 0)
            
            # Extract reasoning tokens from output_tokens_details
            if hasattr(usage, 'output_tokens_details') and usage.output_tokens_details:
                token_usage['reasoning_tokens'] = getattr(usage.output_tokens_details, 'reasoning_tokens', 0)
        
        return token_usage
    
    def _calculate_cost(self, model, token_usage):
        """Calculate cost using LLMPricing"""
        return LLMPricing.calculate_cost(
            model_name=model,
            input_tokens=token_usage.get('input_tokens', 0),
            cached_input_tokens=token_usage.get('cached_tokens', 0),
            output_tokens=token_usage.get('output_tokens', 0),
            reasoning_tokens=token_usage.get('reasoning_tokens', 0)
        )
    
    def _response_to_dict(self, response):
        """Convert response object to dictionary for JSON storage"""
        try:
            return {
                'id': response.id,
                'created_at': response.created_at,
                'model': response.model,
                'status': response.status,
                'usage': {
                    'input_tokens': response.usage.input_tokens if response.usage else 0,
                    'output_tokens': response.usage.output_tokens if response.usage else 0,
                    'total_tokens': response.usage.total_tokens if response.usage else 0,
                } if response.usage else None
            }
        except Exception:
            return {'error': 'Failed to serialize response'}
