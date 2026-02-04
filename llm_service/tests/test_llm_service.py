import os
import json
from unittest import skipUnless
from django.test import TestCase
from django.contrib.auth import get_user_model
from llm_service.services import LLMService
from llm_service.models import LLMCallLog
from llm_service.tools.secret_number import GET_SECRET_NUMBER_TOOL

User = get_user_model()


@skipUnless(
    os.getenv('TEST_APIS') == 'True',
    'API tests disabled - set TEST_APIS=True to enable',
)
class LLMServiceAPITest(TestCase):
    """Test LLM service API calls - only runs when TEST_APIS=True"""
    
    def setUp(self):
        """Set up test data."""
        self.user = User.objects.create_user(
            email="test@example.com",
            password="testpass123",
        )
    
    def test_call_llm_structured_output(self):
        """Test LLMService.call_llm with structured JSON output"""
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
            # Test that we can make a structured call and get a log entry
            call_log = service.call_llm(
                user_prompt=input_text,
                json_schema=json_schema,
                schema_name="text_analysis",
                user=self.user
            )
            
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
            self.assertIn("test_llm_service.py", db_call_log.caller)
            self.assertIn("test_call_llm_structured_output", db_call_log.caller)
            
            # Test raw response persistence
            self.assertIsNotNone(db_call_log.raw_response)
            self.assertIsInstance(db_call_log.raw_response, dict)
            self.assertIn('id', db_call_log.raw_response)
            self.assertIn('status', db_call_log.raw_response)
            self.assertEqual(db_call_log.raw_response['status'], 'completed')
            
            # Test JSON schema persistence
            self.assertIsNotNone(db_call_log.json_schema)
            self.assertIsInstance(db_call_log.json_schema, dict)
            self.assertEqual(db_call_log.json_schema['type'], 'object')
            self.assertIn('properties', db_call_log.json_schema)
            
            # Test parsed JSON persistence
            self.assertIsNotNone(db_call_log.parsed_json)
            self.assertIsInstance(db_call_log.parsed_json, dict)
            self.assertIn('summary', db_call_log.parsed_json)
            self.assertIn('keywords', db_call_log.parsed_json)
            self.assertIn('sentiment', db_call_log.parsed_json)
            self.assertIsInstance(db_call_log.parsed_json['keywords'], list)
            self.assertIn(db_call_log.parsed_json['sentiment'], ['positive', 'negative', 'neutral'])
            
            # Test token usage persistence
            self.assertGreater(db_call_log.input_tokens, 0)
            self.assertGreater(db_call_log.output_tokens, 0)
            self.assertGreater(db_call_log.total_tokens, 0)
            self.assertEqual(db_call_log.total_tokens, db_call_log.input_tokens + db_call_log.output_tokens)
            
            # Test cost calculation persistence
            self.assertIsNotNone(db_call_log.llm_cost_usd)
            self.assertGreater(db_call_log.llm_cost_usd, 0)
            
            # Test timestamps
            self.assertIsNotNone(db_call_log.created_at)
            self.assertIsNotNone(db_call_log.updated_at)
            
            # Test that all fields match between in-memory and database objects
            self.assertEqual(call_log.model, db_call_log.model)
            self.assertEqual(call_log.reasoning_effort, db_call_log.reasoning_effort)
            self.assertEqual(call_log.user_prompt, db_call_log.user_prompt)
            self.assertEqual(call_log.schema_name, db_call_log.schema_name)
            self.assertEqual(call_log.succeeded, db_call_log.succeeded)
            self.assertEqual(call_log.caller, db_call_log.caller)
            self.assertEqual(call_log.input_tokens, db_call_log.input_tokens)
            self.assertEqual(call_log.output_tokens, db_call_log.output_tokens)
            self.assertEqual(call_log.total_tokens, db_call_log.total_tokens)
            self.assertEqual(call_log.llm_cost_usd, db_call_log.llm_cost_usd)
            self.assertEqual(call_log.parsed_json, db_call_log.parsed_json)
            
            # Print for debugging
            # print(f"Call log ID: {call_log.id}")
            # print(f"Caller: {call_log.caller}")
            # print(f"Parsed JSON: {json.dumps(call_log.parsed_json, indent=2)}")
            # print(f"Token usage: {call_log.input_tokens} in, {call_log.output_tokens} out, {call_log.total_tokens} total")
            # print(f"Cost: ${call_log.llm_cost_usd}")
            # print(f"Database verification: ✅ All fields match between in-memory and database objects")
            
        except Exception as e:
            self.fail(f"LLM API call failed: {e}")
    
    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_retry_functionality(self):
        """Test that retry parameter works correctly"""
        service = LLMService()
        service.model = "gpt-5-nano"
        
        # Define a simple JSON schema for testing
        json_schema = {
            "type": "object",
            "properties": {
                "message": {
                    "type": "string",
                    "description": "A simple message"
                }
            },
            "required": ["message"],
            "additionalProperties": False
        }
        
        user_prompt = "Say hello"
        
        try:
            # Test with default retries (2)
            call_log = service.call_llm(
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name="simple_message",
                user=self.user
            )
            
            # Verify it succeeded
            self.assertTrue(call_log.succeeded)
            self.assertIsNotNone(call_log.parsed_json)
            self.assertIn('message', call_log.parsed_json)
            
            # Test with custom retries (0 - no retries)
            call_log_no_retries = service.call_llm(
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name="simple_message",
                retries=0,
                user=self.user
            )
            
            # Verify it still succeeded
            self.assertTrue(call_log_no_retries.succeeded)
            self.assertIsNotNone(call_log_no_retries.parsed_json)
            
            # Test with custom retries (5)
            call_log_many_retries = service.call_llm(
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name="simple_message",
                retries=5,
                user=self.user
            )
            
            # Verify it still succeeded
            self.assertTrue(call_log_many_retries.succeeded)
            self.assertIsNotNone(call_log_many_retries.parsed_json)
            
            # print(f"Retry functionality test: ✅ All retry configurations worked correctly")
            # print(f"Default retries (2): Call log ID {call_log.id}")
            # print(f"No retries (0): Call log ID {call_log_no_retries.id}")
            # print(f"Many retries (5): Call log ID {call_log_many_retries.id}")
            
        except Exception as e:
            self.fail(f"Retry functionality test failed: {e}")
    
    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_failure_handling(self):
        """Test that failed calls return failed log instead of raising exception"""
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
            # This should fail but return a failed log instead of raising an exception
            call_log = service.call_llm(
                user_prompt=user_prompt,
                json_schema=invalid_json_schema,
                schema_name="invalid_schema",
                retries=1,  # Only 1 retry to make it fail faster
                user=self.user
            )
            
            # Verify we got a failed log entry
            self.assertIsNotNone(call_log)
            self.assertIsInstance(call_log, LLMCallLog)
            self.assertFalse(call_log.succeeded)
            self.assertIsNotNone(call_log.raw_response)
            self.assertIn('error', call_log.raw_response)
            self.assertIn('attempts', call_log.raw_response)
            self.assertEqual(call_log.input_tokens, 0)
            self.assertEqual(call_log.output_tokens, 0)
            self.assertEqual(call_log.total_tokens, 0)
            self.assertEqual(call_log.llm_cost_usd, 0)
            self.assertIsNone(call_log.parsed_json)
            
            # Verify it was saved to database
            self.assertIsNotNone(call_log.id)
            self.assertTrue(LLMCallLog.objects.filter(id=call_log.id).exists())
            
            # print(f"Failure handling test: ✅ Failed call returned failed log instead of raising exception")
            # print(f"Failed log ID: {call_log.id}")
            # print(f"Error details: {call_log.raw_response}")
            
        except Exception as e:
            # If we get here, it means the method raised an exception instead of returning failed log
            self.fail(f"Expected failed log return, but got exception: {e}")
    
    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_with_tool_calling(self):
        """Test that tool calling works correctly"""
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
            # Test tool calling with required tool choice
            call_log = service.call_llm(
                user_prompt=user_prompt,
                json_schema=json_schema,
                schema_name="secret_number_response",
                tools=[GET_SECRET_NUMBER_TOOL],
                system_instructions="You are a helpful assistant that can use tools to get information.",
                reasoning_effort="medium",
                retries=1,  # Reduce retries for faster testing
                user=self.user
            )
            
            # Verify we got a successful log entry
            self.assertIsNotNone(call_log)
            self.assertIsInstance(call_log, LLMCallLog)
            self.assertTrue(call_log.succeeded)
            
            # Verify the parsed JSON contains the secret number
            self.assertIsNotNone(call_log.parsed_json)
            self.assertIn('secret_number', call_log.parsed_json)
            self.assertIn('message', call_log.parsed_json)
            self.assertEqual(call_log.parsed_json['secret_number'], 9999)
            
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
            
            # print(f"Tool calling test: ✅ Secret number {call_log.parsed_json['secret_number']} found in response")
            # print(f"Message: {call_log.parsed_json['message']}")
            # print(f"Call log ID: {call_log.id}")
            
        except Exception as e:
            self.fail(f"Tool calling test failed: {e}")

    @skipUnless(os.getenv('TEST_APIS') == 'True', 'API tests disabled - set TEST_APIS=True to enable')
    def test_call_llm_stream_structured_output(self):
        """Test LLMService.call_llm_stream yields events and a final call_log"""
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
        self.assertIn("response", final_data)
        call_log = final_data["call_log"]
        self.assertIsInstance(call_log, LLMCallLog)
        self.assertTrue(call_log.succeeded)
        self.assertIsNotNone(call_log.parsed_json)
        self.assertIn("message", call_log.parsed_json)
        self.assertGreater(len(events), 0)
        self.assertTrue(
            any("output_text" in t or "completed" in t for t in events),
            f"Expected output or completed events, got: {events[:5]}",
        )
