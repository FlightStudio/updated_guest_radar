web: gunicorn app:app
worker: celery -A app.celery_app worker --loglevel=info