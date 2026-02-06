"""
Fallback pricing when LiteLLM does not return cost (e.g. new or custom models).
USD per 1M tokens; used only for allowed models when cost is missing.
"""
from decimal import Decimal
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

# model string or price_plan -> { "input": USD per 1M tokens, "output": USD per 1M tokens }
# Add entries for LLM_ALLOWED_MODELS as needed.
FALLBACK_PRICING: dict[str, dict[str, Decimal]] = {
    "openai/gpt-4o": {"input": Decimal("2.50"), "output": Decimal("10.00")},
    "openai/gpt-4o-mini": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "openai/gpt-5.2": {"input": Decimal("2.50"), "output": Decimal("10.00")},
    "openai/gpt-5-mini": {"input": Decimal("0.40"), "output": Decimal("1.60")},
    "openai/gpt-5-nano": {"input": Decimal("0.15"), "output": Decimal("0.60")},
    "anthropic/claude-3-5-sonnet-20241022": {"input": Decimal("3.00"), "output": Decimal("15.00")},
}


def get_fallback_cost_usd(
    model: str,
    input_tokens: int = 0,
    output_tokens: int = 0,
) -> Decimal | None:
    """Compute cost from fallback table if model is present. Returns None if no price."""
    pricing = FALLBACK_PRICING.get(model)
    if not pricing:
        return None
    input_per_m = Decimal(input_tokens) / Decimal(1_000_000)
    output_per_m = Decimal(output_tokens) / Decimal(1_000_000)
    cost = input_per_m * pricing["input"] + output_per_m * pricing["output"]
    return cost.quantize(Decimal("0.00000001"))
