FROM --platform=$BUILDPLATFORM python:3.10.12-alpine3.18 AS base
WORKDIR /home/django/backend
RUN adduser -D django
RUN apk update && apk add --no-cache netcat-openbsd

COPY --chown=django . /home/django/backend
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

USER django
RUN  python3 -m venv /home/django/venv && \
     pip3 install -r requirements.txt --no-cache-dir
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]

FROM base as dev
CMD python3 manage.py runserver 0.0.0.0:8000

FROM base AS prod
CMD python3 -m gunicorn backend.wsgi:application --bind 0.0.0.0:8000
