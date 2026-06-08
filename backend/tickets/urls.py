# backend/tickets/urls.py
"""URL routes for the tickets app."""

from django.urls import path
from . import views

app_name = 'tickets'

urlpatterns = [
    path('', views.tickets_page, name='tickets'),
]
