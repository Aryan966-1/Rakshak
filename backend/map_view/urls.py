# backend/map_view/urls.py
"""URL routes for the map view app."""

from django.urls import path
from . import views

app_name = 'map_view'

urlpatterns = [
    path('', views.map_page, name='map'),
]
