#!/usr/bin/env bash

# El directorio de trabajo se establece en la raíz del proyecto en Render
# Por lo tanto, le decimos a Gunicorn que el módulo 'app' está dentro de la carpeta 'backend'
gunicorn --chdir backend app:app --bind 0.0.0.0:$PORT