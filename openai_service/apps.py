from django.apps import AppConfig


class OpenaiServiceConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'openai_service'
    label = 'llm_service'  # Keep for DB/migrations compatibility
