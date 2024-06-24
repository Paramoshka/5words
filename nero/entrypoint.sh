#!/bin/sh

echo 'Waiting for postgres...'

while ! nc -z $POSTGRES_HOST 5432; do
    sleep 0.1
done

echo 'PostgreSQL started'

echo 'Running migrations...'
python manage.py migrate

#echo 'create superuser'
#echo "from django.contrib.auth import get_user_model; User = get_user_model(); User.objects.create_superuser('admin', 'admin@admin.com', 'admin')" | python manage.py shell

#echo 'Collecting static files...'
#python manage.py collectstatic --no-input

exec "$@"