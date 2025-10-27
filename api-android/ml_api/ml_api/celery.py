# ml_api/celery.py

import os
from celery import Celery

# Establece la variable de entorno para Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'ml_api.settings')

# Crea la aplicación Celery
app = Celery('ml_api')

# Carga la configuración de Django (necesita el prefijo CELERY_)
app.config_from_object('django.conf:settings', namespace='CELERY')

# Auto-descubre las tareas de las apps de Django
app.autodiscover_tasks()