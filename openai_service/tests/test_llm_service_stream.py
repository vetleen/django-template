import os
import json
from unittest import skipUnless
from django.test import TestCase
from django.contrib.auth import get_user_model
from openai_service.services import LLMService
from openai_service.models import LLMCallLog
from openai_service.tools.secret_number import GET_SECRET_NUMBER_TOOL

User = get_user_model()


@skipUnless(
    os.getenv('TEST_APIS') == 'True',
    'API tests disabled - set TEST_APIS=True to enable',
)
class LLMServiceStreamAPITest(TestCase):
    """Test LLM service streaming API calls - only runs when TEST_APIS=True"""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            email="test@example.com",
            password="testpass123",
        )
    
    def test_call_llm_stream_structured_output(self):
        """Test LLMService.call_llm_stream with structured JSON output"""
        service = LLMService()
        service.model = "gpt-5-nano"
        
        # Define a complex JSON schema for testing
        json_schema = {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "A brief summary of the input"
                },
                "keywords": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Key terms from the input"
                },
                "sentiment": {
                    "type": "string",
                    "enum": ["positive", "negative", "neutral"],
                    "description": "The sentiment of the input"
                }
            },
            "required": ["summary", "keywords", "sentiment"],
            "additionalProperties": False
        }
        
        input_text = "I love this new product! It's amazing and works perfectly."
        
        try:
            # Collect events and final data
            events = []
            event_types = []
            final_data = None
            
            gen = service.call_llm_stream(
                user_prompt=input_text,
                json_schema=json_schema,
                schema_name="text_analysis",
                user=self.user
            )
            
            for event_type, event in gen:
                if event_type == "final":
                    final_data = event
                    break
                events.append((event_type, event))
                event_types.append(event_type)
            
            # Test that we got final data
            self.assertIsNotNone(final_data)
            self.assertIn("call_log", final_data)
            self.assertIn("response", final_data)
            
            call_log = final_data["call_log"]
            response = final_data["response"]
            
            # Test that we got a LLMCallLog object
            self.assertIsNotNone(call_log)
            self.assertIsInstance(call_log, LLMCallLog)
            
            # Test basic log fields
            self.assertEqual(call_log.model, "gpt-5-nano")
            self.assertEqual(call_log.reasoning_effort, "low")
            self.assertEqual(call_log.user_prompt, input_text)
            self.assertEqual(call_log.schema_name, "text_analysis")
            self.assertTrue(call_log.succeeded)
            
            # Test caller information
            self.assertIsNotNone(call_log.caller)
            self.assertNotEqual(call_log.caller, "unknown")
            
            # Test raw response
            self.assertIsNotNone(call_log.raw_response)
            self.assertIn('id', call_log.raw_response)
            self.assertIn('status', call_log.raw_response)
            
            # Test parsed JSON
            self.assertIsNotNone(call_log.parsed_json)
            self.assertIn('summary', call_log.parsed_json)
            self.assertIn('keywords', call_log.parsed_json)
            self.assertIn('sentiment', call_log.parsed_json)
            self.assertIsInstance(call_log.parsed_json['keywords'], list)
            self.assertIn(call_log.parsed_json['sentiment'], ['positive', 'negative', 'neutral'])
            
            # Test token usage
            self.assertGreater(call_log.input_tokens, 0)
            self.assertGreater(call_log.output_tokens, 0)
            self.assertGreater(call_log.total_tokens, 0)
            
            # Test cost calculation
            self.assertIsNotNone(call_log.llm_cost_usd)
            self.assertGreater(call_log.llm_cost_usd, 0)
            
            # Test that we received stream events
            self.assertGreater(len(events), 0)
            self.assertTrue(
                any("output_text" in t or "completed" in t for t in event_types),
                f"Expected output or completed events, got: {event_types[:10]}"
            )
            
            # Test database storage - verify object exists
            self.assertIsNotNone(call_log.id)
            self.assertTrue(LLMCallLog.objects.filter(id=call_log.id).exists())
            
            # Retrieve the object from database to verify persistence
            db_call_log = LLMCallLog.objects.get(id=call_log.id)
            
            # Test all fields were saved correctly
            self.assertEqual(db_call_log.model, "gpt-5-nano")
            self.assertEqual(db_call_log.reasoning_effort, "low")
            self.assertEqual(db_call_log.user_prompt, input_text)
            self.assertEqual(db_call_log.schema_name, "text_analysis")
            self.assertTrue(db_call_log.succeeded)
            
            # Test caller information persistence
            self.assertIsNotNone(db_call_log.caller)
            self.assertNotEqual(db_call_log.caller, "unknown")
            self.assertIn("test_llm_service_stream.py", db_call_log.caller)
            self.assertIn("test_call_llm_stream_structured_output", db_call_log.caller)
            
            # Test parsed JSON persistence
            self.assertIsNotNone(db_call_log.parsed_json)
            self.assertIsInstance(db_call_log.parsed_json, dict)
            self.assertIn('summary', db_call_log.parsed_json)
            self.assertIn('keywords', db_call_log.parsed_json)
            self.assertIn('sentiment', db_call_log.parsed_json)
            
            # Test that all fields match between in-memory and database objects
            self.assertEqual(call_log.model, db_call_log.model)
            self.assertEqual(call_log.reasoning_effort, db_call_log.reasoning_effort)
            self.assertEqual(call_log.user_prompt, db_call_log.user_prompt)
            self.assertEqual(call_log.schema_name, db_call_log.schema_name)
            self.assertEqual(call_log.succeeded, db_call_log.succeeded)
            self.assertEqual(call_log.parsed_json, db_call_log.parsed_json)
            
        except Exception as e:
            self.fail(f"LLM streaming API call failed: {e}")
    
    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_stream_simple_message(self):
        """Test LLMService.call_llm_stream with a simple message schema"""
        service = LLMService()
        service.model = "gpt-5-nano"
        
        json_schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "A short message"},
            },
            "required": ["message"],
            "additionalProperties": False,
        }
        
        events = []
        final_data = None
        gen = service.call_llm_stream(
            user_prompt="Say hello in one word.",
            json_schema=json_schema,
            schema_name="simple_message",
            user=self.user,
        )
        for event_type, event in gen:
            if event_type == "final":
                final_data = event
                break
            events.append(event_type)
        
        self.assertIsNotNone(final_data)
        self.assertIn("call_log", final_data)
        call_log = final_data["call_log"]
        self.assertIsInstance(call_log, LLMCallLog)
        self.assertTrue(call_log.succeeded)
        self.assertIsNotNone(call_log.parsed_json)
        self.assertIn("message", call_log.parsed_json)
        self.assertGreater(len(events), 0)
    
    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_stream_with_tool_calling(self):
        """Test that tool calling works correctly with streaming"""
        service = LLMService()
        service.model = "gpt-5-nano"
        
        # Define a JSON schema that expects the secret number
        json_schema = {
            "type": "object",
            "properties": {
                "secret_number": {
                    "type": "number",
                    "description": "The secret number returned by the tool"
                },
                "message": {
                    "type": "string",
                    "description": "A message about the secret number"
                }
            },
            "required": ["secret_number", "message"],
            "additionalProperties": False
        }
        
        user_prompt = "Please get the secret number and tell me what it is."
        
        try:
            events = []
            event_types = []
            function_call_events = []
            final_data = None
            
            gen = service.call_llm_stream(
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name="secret_number_response",
                tools=[GET_SECRET_NUMBER_TOOL],
                system_instructions="You are a helpful assistant that can use tools to get information.",
                reasoning_effort="medium",
                user=self.user
            )
            
            for event_type, event in gen:
                if event_type == "final":
                    final_data = event
                    break
                events.append((event_type, event))
                event_types.append(event_type)
                
                # Track function call related events
                if "function_call" in event_type or "output_item" in event_type:
                    function_call_events.append(event_type)
            
            # Verify we got a successful log entry
            self.assertIsNotNone(final_data)
            self.assertIn("call_log", final_data)
            call_log = final_data["call_log"]
            self.assertIsNotNone(call_log)
            self.assertIsInstance(call_log, LLMCallLog)
            self.assertTrue(call_log.succeeded)
            
            # Verify the parsed JSON contains the secret number
            self.assertIsNotNone(call_log.parsed_json)
            self.assertIn('secret_number', call_log.parsed_json)
            self.assertIn('message', call_log.parsed_json)
            self.assertEqual(call_log.parsed_json['secret_number'], 9999)
            
            # Verify we received function call events (immediate handling)
            self.assertGreater(len(function_call_events), 0, 
                             "Expected function call events during streaming")
            
            # Verify database storage
            self.assertIsNotNone(call_log.id)
            self.assertTrue(LLMCallLog.objects.filter(id=call_log.id).exists())
            
            # Verify token usage (should be higher due to tool calling)
            self.assertGreater(call_log.input_tokens, 0)
            self.assertGreater(call_log.output_tokens, 0)
            self.assertGreater(call_log.total_tokens, 0)
            
            # Verify cost calculation
            self.assertIsNotNone(call_log.llm_cost_usd)
            self.assertGreater(call_log.llm_cost_usd, 0)
            
        except Exception as e:
            self.fail(f"Tool calling streaming test failed: {e}")
    
    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_stream_failure_handling(self):
        """Test that failed streams return failed log instead of raising exception"""
        service = LLMService()
        service.model = "gpt-5-nano"
        
        # Define a JSON schema that will cause an error (invalid schema)
        invalid_json_schema = {
            "type": "invalid_type",  # This should cause an API error
            "properties": {
                "message": {
                    "type": "string"
                }
            }
        }
        
        user_prompt = "Say hello"
        
        try:
            events = []
            final_data = None
            
            gen = service.call_llm_stream(
                user_prompt=user_prompt,
                json_schema=invalid_json_schema,
                schema_name="invalid_schema",
                user=self.user
            )
            
            for event_type, event in gen:
                if event_type == "final":
                    final_data = event
                    break
                events.append(event_type)
            
            # Verify we got a failed log entry
            self.assertIsNotNone(final_data)
            self.assertIn("call_log", final_data)
            call_log = final_data["call_log"]
            self.assertIsNotNone(call_log)
            self.assertIsInstance(call_log, LLMCallLog)
            self.assertFalse(call_log.succeeded)
            self.assertIsNotNone(call_log.raw_response)
            self.assertIn('error', call_log.raw_response)
            self.assertEqual(call_log.input_tokens, 0)
            self.assertEqual(call_log.output_tokens, 0)
            self.assertEqual(call_log.total_tokens, 0)
            self.assertEqual(call_log.llm_cost_usd, 0)
            self.assertIsNone(call_log.parsed_json)
            
            # Verify it was saved to database
            self.assertIsNotNone(call_log.id)
            self.assertTrue(LLMCallLog.objects.filter(id=call_log.id).exists())
            
        except Exception as e:
            # If we get here, it means the method raised an exception instead of returning failed log
            self.fail(f"Expected failed log return, but got exception: {e}")
    
    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_stream_event_types(self):
        """Test that we receive expected event types during streaming"""
        service = LLMService()
        service.model = "gpt-5-nano"
        
        json_schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "A message"},
            },
            "required": ["message"],
            "additionalProperties": False,
        }
        
        event_types = []
        final_data = None
        
        gen = service.call_llm_stream(
            user_prompt="Say hello",
            json_schema=json_schema,
            schema_name="test_schema",
            user=self.user,
        )
        
        for event_type, event in gen:
            if event_type == "final":
                final_data = event
                break
            event_types.append(event_type)
        
        # Verify we got final data
        self.assertIsNotNone(final_data)
        
        # Verify we received some events
        self.assertGreater(len(event_types), 0)
        
        # Verify we got at least one output-related event
        has_output_event = any(
            "output_text" in t or "completed" in t or "refusal" in t 
            for t in event_types
        )
        self.assertTrue(
            has_output_event,
            f"Expected output/completed/refusal events, got: {event_types}"
        )
    
    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_stream_usage_tracking(self):
        """Test that usage tracking works correctly with streaming"""
        service = LLMService()
        service.model = "gpt-5-nano"
        
        json_schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
            },
            "required": ["message"],
            "additionalProperties": False,
        }
        
        from openai_service.models import UserMonthlyUsage
        from datetime import datetime
        
        # Get initial usage count
        now = datetime.now()
        initial_usage = UserMonthlyUsage.objects.filter(
            user=self.user,
            year=now.year,
            month=now.month
        ).first()
        
        initial_calls = initial_usage.total_calls if initial_usage else 0
        initial_tokens = initial_usage.total_tokens if initial_usage else 0
        
        # Make streaming call
        gen = service.call_llm_stream(
            user_prompt="Test message",
            json_schema=json_schema,
            schema_name="test",
            user=self.user,
        )
        
        # Consume generator
        final_data = None
        for event_type, event in gen:
            if event_type == "final":
                final_data = event
                break
        
        self.assertIsNotNone(final_data)
        call_log = final_data["call_log"]
        self.assertTrue(call_log.succeeded)
        
        # Verify usage was tracked
        usage = UserMonthlyUsage.objects.get(
            user=self.user,
            year=now.year,
            month=now.month
        )
        
        self.assertEqual(usage.total_calls, initial_calls + 1)
        self.assertGreater(usage.total_tokens, initial_tokens)
        self.assertGreater(usage.total_cost_usd, 0)
    
    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_stream_retry_parameter(self):
        """Test that retry parameter works correctly with streaming"""
        service = LLMService()
        service.model = "gpt-5-nano"
        
        json_schema = {
            "type": "object",
            "properties": {
                "message": {"type": "string", "description": "A simple message"},
            },
            "required": ["message"],
            "additionalProperties": False,
        }
        
        try:
            # Test with default retries (2)
            events = []
            final_data = None
            gen = service.call_llm_stream(
                user_prompt="Say hello",
                json_schema=json_schema,
                schema_name="simple_message",
                user=self.user,
            )
            for event_type, event in gen:
                if event_type == "final":
                    final_data = event
                    break
                events.append(event_type)
            
            self.assertIsNotNone(final_data)
            call_log = final_data["call_log"]
            self.assertTrue(call_log.succeeded)
            self.assertIsNotNone(call_log.parsed_json)
            
            # Test with custom retries (0 - no retries)
            gen_no_retries = service.call_llm_stream(
                user_prompt="Say hello",
                json_schema=json_schema,
                schema_name="simple_message",
                retries=0,
                user=self.user,
            )
            final_data_no_retries = None
            for event_type, event in gen_no_retries:
                if event_type == "final":
                    final_data_no_retries = event
                    break
            
            self.assertIsNotNone(final_data_no_retries)
            call_log_no_retries = final_data_no_retries["call_log"]
            self.assertTrue(call_log_no_retries.succeeded)
            
            # Test with custom retries (5)
            gen_many_retries = service.call_llm_stream(
                user_prompt="Say hello",
                json_schema=json_schema,
                schema_name="simple_message",
                retries=5,
                user=self.user,
            )
            final_data_many_retries = None
            for event_type, event in gen_many_retries:
                if event_type == "final":
                    final_data_many_retries = event
                    break
            
            self.assertIsNotNone(final_data_many_retries)
            call_log_many_retries = final_data_many_retries["call_log"]
            self.assertTrue(call_log_many_retries.succeeded)
            
        except Exception as e:
            self.fail(f"Retry functionality streaming test failed: {e}")
