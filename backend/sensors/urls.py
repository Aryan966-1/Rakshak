# backend/sensors/urls.py
"""URL routes for the sensors app (Dashboard)."""

from django.urls import path
from . import views

app_name = 'sensors'

urlpatterns = [
    path('', views.dashboard, name='dashboard'),
]
