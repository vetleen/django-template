"""
Secret number tool for LLM function calling.
"""
from openai_service.tools.registry import tool_registry


def get_secret_number():
    """Returns a secret number."""
    return 9999


# Tool definition for LLM
GET_SECRET_NUMBER_TOOL = {
    "type": "function",
    "name": "get_secret_number",
    "description": "Returns a secret number",
    "parameters": {
        "type": "object",
        "properties": {},
        "required": [],
        "additionalProperties": False
    },
    "strict": True
}

# Register the tool
tool_registry.register("get_secret_number", get_secret_number)
