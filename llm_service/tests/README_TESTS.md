# llm_service test suite

Run all llm_service tests (no API keys required; LiteLLM is mocked):

```bash
python manage.py test llm_service
```

## Layout

- **test_conf.py** – Configuration: `get_default_model`, `get_allowed_models`, `is_model_allowed`, timeouts, retries, hooks.
- **test_request_result.py** – `LLMRequest` / `LLMResult`: `to_completion_kwargs`, usage properties, `succeeded`.
- **test_pricing.py** – Fallback pricing: `get_fallback_cost_usd`, `FALLBACK_PRICING`.
- **test_models.py** – `LLMCallLog`: minimal/full creation, status choices, `LOGGING_FAILED`.
- **test_client.py** – `completion()`, `get_client()`: model validation, pre/post hooks, success/error logging, metadata, streaming (one log when stream ends).
- **test_litellm_client.py** – `LiteLLMClient`: timeout passed to litellm, `completion`/`acompletion` delegation.
- **test_integration.py** – Full flow with mocked `litellm.completion`: log fields, user/metadata, stream single log, minimal log fallback when `_write_log` fails.
- **test_live_api.py** – **Live API tests** (run only when `TEST_APIS=True`): real calls to `openai/gpt-5-nano` for basic completion, structured JSON response, streaming, and tool use. Skipped otherwise.

All tests use mocks except **test_live_api.py**, which runs only when `TEST_APIS=True` (and requires `OPENAI_API_KEY` and `openai/gpt-5-nano` in `LLM_ALLOWED_MODELS`):

```bash
TEST_APIS=True python manage.py test llm_service.tests.test_live_api -v 2
```
