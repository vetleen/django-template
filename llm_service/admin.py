from django.contrib import admin

from .models import LLMCallLog, UserMonthlyUsage


@admin.register(LLMCallLog)
class LLMCallLogAdmin(admin.ModelAdmin):
    list_display = ("id", "model", "caller", "succeeded", "total_tokens", "llm_cost_usd", "created_at")
    list_filter = ("succeeded", "model", "created_at")
    search_fields = ("model", "caller")
    ordering = ("-created_at",)


@admin.register(UserMonthlyUsage)
class UserMonthlyUsageAdmin(admin.ModelAdmin):
    list_display = ("id", "user", "year", "month", "total_calls", "total_tokens", "total_cost_usd")
    list_filter = ("year", "month")
    search_fields = ("user__email", "user__username")
    ordering = ("-year", "-month")
