"""
LLM providers: one module per provider. The router resolves model -> provider via registry.
"""

from llm_service.registry import get_provider_key, is_supported, list_supported_models
from llm_service.providers.openai import OpenAIProvider


def get_provider(model: str | None):
    """
    Return the provider instance for the given model.
    Model must be in the supported registry; raises ValueError if not.
    """
    if model is None:
        raise ValueError("model is required")
    if not is_supported(model):
        raise ValueError(f"Unsupported model: {model}. Supported: {list_supported_models()}")
    key = get_provider_key(model)
    if key == "openai":
        return OpenAIProvider()
    raise ValueError(f"Unknown provider key for model {model}: {key}")
