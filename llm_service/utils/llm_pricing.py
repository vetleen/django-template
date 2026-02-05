"""
LLM pricing utilities for cost calculation.
"""

from decimal import Decimal
from typing import Dict, Optional


class LLMPricing:
    """Handles LLM model pricing and cost calculations."""
    
    # Pricing data in USD per 1M tokens (Standard tier, from platform.openai.com/docs/pricing)
    PRICING_DATA = {
        'gpt-5.2': {
            'input': Decimal('1.75'),
            'cached_input': Decimal('0.175'),
            'output': Decimal('14.00')
        },
        'gpt-5': {
            'input': Decimal('1.25'),
            'cached_input': Decimal('0.125'),
            'output': Decimal('10.00')
        },
        'gpt-5-mini': {
            'input': Decimal('0.25'),
            'cached_input': Decimal('0.025'),
            'output': Decimal('2.00')
        },
        'gpt-5-nano': {
            'input': Decimal('0.05'),
            'cached_input': Decimal('0.005'),
            'output': Decimal('0.40')
        },
        'gpt-4.1': {
            'input': Decimal('2.00'),
            'cached_input': Decimal('0.50'),
            'output': Decimal('8.00')
        },
        # Legacy/alias plans (not in registry)
        'gpt-5-chat-latest': {
            'input': Decimal('1.25'),
            'cached_input': Decimal('0.125'),
            'output': Decimal('10.00')
        },
        'gpt-4o': {
            'input': Decimal('2.50'),
            'cached_input': Decimal('1.25'),
            'output': Decimal('10.00')
        }
    }
    
    @classmethod
    def get_model_pricing(cls, model_name: str) -> Optional[Dict[str, Decimal]]:
        """
        Get pricing data for a specific model.
        
        Args:
            model_name: Name of the model (e.g., 'gpt-5', 'gpt-4o')
            
        Returns:
            Dict with pricing data or None if model not found
        """
        return cls.PRICING_DATA.get(model_name)
    
    @classmethod
    def calculate_cost(
        cls,
        model_name: str,
        input_tokens: int = 0,
        cached_input_tokens: int = 0,
        output_tokens: int = 0,
        reasoning_tokens: int = 0
    ) -> Decimal:
        """
        Calculate the cost for an LLM API call.
        
        Args:
            model_name: Name of the model used
            input_tokens: Number of input tokens
            cached_input_tokens: Number of cached input tokens
            output_tokens: Number of output tokens
            reasoning_tokens: Number of reasoning tokens (uses output pricing)
            
        Returns:
            Total cost in USD as Decimal
        """
        pricing = cls.get_model_pricing(model_name)
        if not pricing:
            # Default to gpt-5 pricing if model not found
            pricing = cls.PRICING_DATA['gpt-5']
        
        # Convert token counts to millions for pricing calculation
        input_cost = (Decimal(input_tokens) / Decimal('1000000')) * pricing['input']
        cached_input_cost = (Decimal(cached_input_tokens) / Decimal('1000000')) * pricing['cached_input']
        output_cost = (Decimal(output_tokens) / Decimal('1000000')) * pricing['output']
        reasoning_cost = (Decimal(reasoning_tokens) / Decimal('1000000')) * pricing['output']
        
        total_cost = input_cost + cached_input_cost + output_cost + reasoning_cost
        
        # Round to 6 decimal places for precision
        return total_cost.quantize(Decimal('0.000001'))
    
    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names."""
        return list(cls.PRICING_DATA.keys())
