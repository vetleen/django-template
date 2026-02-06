from django.contrib import admin
from django.utils.html import format_html

from .models import LLMCallLog


@admin.register(LLMCallLog)
class LLMCallLogAdmin(admin.ModelAdmin):
    list_display = (
        "id",
        "model",
        "status",
        "total_tokens",
        "cost_usd",
        "duration_ms",
        "created_at",
        "user",
        "request_id",
    )
    list_filter = ("status", "model", "is_stream")
    search_fields = ("request_id", "model", "error_message")
    readonly_fields = (
        "id",
        "created_at",
        "model",
        "is_stream",
        "request_kwargs",
        "prompt_hash",
        "prompt_preview",
        "provider_response_id",
        "response_model",
        "response_preview",
        "response_hash",
        "input_tokens",
        "output_tokens",
        "total_tokens",
        "cost_usd",
        "cost_source",
        "status",
        "error_type",
        "error_message",
        "http_status",
        "retry_count",
        "provider_request_id",
        "metadata",
        "request_id",
        "duration_ms",
        "user",
    )
    fieldsets = (
        (None, {"fields": ("id", "created_at", "duration_ms", "status", "user", "request_id", "metadata")}),
        ("Request", {"fields": ("model", "is_stream", "request_kwargs", "prompt_hash", "prompt_preview")}),
        ("Response", {"fields": ("provider_response_id", "response_model", "response_preview", "response_hash")}),
        ("Usage / cost", {"fields": ("input_tokens", "output_tokens", "total_tokens", "cost_usd", "cost_source")}),
        ("Errors", {"fields": ("error_type", "error_message", "http_status", "retry_count", "provider_request_id")}),
    )
    ordering = ["-created_at"]
    date_hierarchy = "created_at"
