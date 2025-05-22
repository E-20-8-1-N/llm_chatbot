#!/bin/sh

# Exit on error
set -e

echo "Running Django migrations..."
python manage.py migrate --noinput

echo "Starting Gunicorn..."
# Adjust workers as needed. For an iGPU and LLM, 1 worker might be best to avoid VRAM contention.
# If not using GPU heavily or for other tasks, 2-3 might be okay.
# The default Gunicorn port is 8000.
exec gunicorn chatbot_project.wsgi:application --bind 0.0.0.0:8000 --workers 1 --timeout 120