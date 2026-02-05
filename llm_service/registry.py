"""
Supported models: model id -> provider and price plan.
Single source of truth for "what we support" and routing.
"""

# model_id -> { "provider": str, "price_plan": str }
# price_plan keys into LLMPricing.PRICING_DATA (USD per 1M tokens)
SUPPORTED_MODELS = {
    "gpt-5.2": {"provider": "openai", "price_plan": "gpt-5.2"},
    "gpt-5": {"provider": "openai", "price_plan": "gpt-5"},
    "gpt-5-mini": {"provider": "openai", "price_plan": "gpt-5-mini"},
    "gpt-5-nano": {"provider": "openai", "price_plan": "gpt-5-nano"},
    "gpt-4.1": {"provider": "openai", "price_plan": "gpt-4.1"},
}


def get_provider_key(model: str | None) -> str | None:
    """Return provider key for the model, or None if not supported."""
    if model is None:
        return None
    entry = SUPPORTED_MODELS.get(model)
    return entry["provider"] if entry else None


def get_price_plan(model: str | None) -> str | None:
    """Return price plan key for the model (for LLMPricing), or None if not supported."""
    if model is None:
        return None
    entry = SUPPORTED_MODELS.get(model)
    return entry["price_plan"] if entry else None


def list_supported_models():
    """Return list of supported model ids (order preserved)."""
    return list(SUPPORTED_MODELS.keys())


def is_supported(model: str | None) -> bool:
    """Return True if the model is in the supported list."""
    return model is not None and model in SUPPORTED_MODELS
