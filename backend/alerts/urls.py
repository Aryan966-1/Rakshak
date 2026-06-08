# backend/alerts/urls.py
"""URL routes for the alerts app."""

from django.urls import path
from . import views

app_name = 'alerts'

urlpatterns = [
    path('', views.alerts_page, name='alerts'),
]
