# backend/rakshak_project/wsgi.py
"""WSGI config for Rakshak project."""

import os
from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'rakshak_project.settings')
application = get_wsgi_application()
