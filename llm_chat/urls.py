from django.urls import path

from . import views

urlpatterns = [
    path("", views.chat_view, name="chat"),
    path("<uuid:thread_id>/", views.chat_view, name="chat_thread"),
    path("api/threads/<uuid:thread_id>/messages/", views.chat_messages_json, name="chat_messages_json"),
]
