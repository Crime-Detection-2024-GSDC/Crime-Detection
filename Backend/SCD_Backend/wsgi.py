"""
WSGI config for SCD_Backend project.

It exposes the WSGI callable as a module-level variable named ``application``.

For more information on this file, see
https://docs.djangoproject.com/en/4.2/howto/deployment/wsgi/
"""

import os

from django.core.wsgi import get_wsgi_application

os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'SCD_Backend.settings')
os.environ.setdefault('RUN_CRON', "true")

application = get_wsgi_application()