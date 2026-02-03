"""
Test package for llm_chat.

We keep API-hitting tests behind the same TEST_APIS flag used by openai_service,
but most chat tests mock LLMService / Channels so they run fast and offline.
"""

