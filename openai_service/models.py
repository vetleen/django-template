from django.db import models
from django.contrib.auth import get_user_model
from decimal import Decimal
import json


class LLMCallLog(models.Model):
    """
    Logs all LLM API calls for tracking, debugging, and cost analysis.
    """
    
    # Raw response data
    raw_response = models.JSONField(help_text="Complete raw response from LLM API")
    
    # Call context
    caller = models.CharField(max_length=500, help_text="File and function where call_llm was invoked")
    model = models.CharField(max_length=100, help_text="LLM model used")
    reasoning_effort = models.CharField(max_length=20, default="low", help_text="Reasoning effort level")
    
    # Input/Output data
    system_instructions = models.TextField(blank=True, null=True, help_text="System instructions sent to LLM")
    user_prompt = models.TextField(blank=True, null=True, help_text="User prompt sent to LLM")
    
    # Structured output data
    json_schema = models.JSONField(blank=True, null=True, help_text="JSON schema used for structured output")
    schema_name = models.CharField(max_length=100, blank=True, null=True, help_text="Name of the schema used")
    parsed_json = models.JSONField(blank=True, null=True, help_text="Parsed JSON data from structured output")
    
    # Token usage
    input_tokens = models.PositiveIntegerField(default=0, help_text="Number of input tokens")
    output_tokens = models.PositiveIntegerField(default=0, help_text="Number of output tokens")
    cached_tokens = models.PositiveIntegerField(default=0, help_text="Number of cached input tokens")
    reasoning_tokens = models.PositiveIntegerField(default=0, help_text="Number of reasoning tokens")
    total_tokens = models.PositiveIntegerField(default=0, help_text="Total tokens used")
    
    # Status and cost
    succeeded = models.BooleanField(default=True, help_text="Whether the LLM call succeeded")
    llm_cost_usd = models.DecimalField(
        max_digits=10, 
        decimal_places=6, 
        default=Decimal('0.000000'),
        help_text="Cost of this LLM call in USD"
    )
    response_time_seconds = models.FloatField(
        null=True,
        blank=True,
        help_text="Response time in seconds"
    )
    
    # Timestamps
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
        verbose_name = "LLM Call Log"
        verbose_name_plural = "LLM Call Logs"
    
    def __str__(self):
        return f"LLM Call {self.id} - {self.model} ({self.caller})"
    

class UserMonthlyUsage(models.Model):
    """Tracks monthly LLM usage and costs per user."""
    
    user = models.ForeignKey(
        get_user_model(),
        on_delete=models.CASCADE,
        related_name='monthly_usage',
        help_text="The user this usage record belongs to"
    )
    
    year = models.PositiveIntegerField(
        help_text="Year (e.g., 2025)"
    )
    
    month = models.PositiveIntegerField(
        help_text="Month (1-12)"
    )
    
    total_cost_usd = models.DecimalField(
        max_digits=10,
        decimal_places=6,
        default=Decimal('0.000000'),
        help_text="Total LLM cost in USD for this month"
    )
    
    total_tokens = models.PositiveIntegerField(
        default=0,
        help_text="Total tokens used this month"
    )
    
    total_calls = models.PositiveIntegerField(
        default=0,
        help_text="Total LLM calls made this month"
    )
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        unique_together = ['user', 'year', 'month']
        ordering = ['-year', '-month']
        verbose_name = "User Monthly Usage"
        verbose_name_plural = "User Monthly Usage"
        indexes = [
            models.Index(fields=['user', 'year', 'month']),
            models.Index(fields=['year', 'month']),
        ]
    
    def __str__(self):
        return f"{self.user.email} - {self.year}-{self.month:02d} (${self.total_cost_usd})"
    
    @property
    def month_name(self):
        """Return the month name."""
        from datetime import datetime
        return datetime(self.year, self.month, 1).strftime('%B')
    
    @property
    def period_display(self):
        """Return formatted period string."""
        return f"{self.month_name} {self.year}"


