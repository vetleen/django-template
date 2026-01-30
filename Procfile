web: daphne -b 0.0.0.0 -p $PORT config.asgi:application
release: npm run build:css && python manage.py migrate --noinput && python manage.py compress --force && python manage.py collectstatic --noinput
