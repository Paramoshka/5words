FROM --platform=$BUILDPLATFORM python:3.10.12-bookworm AS base
WORKDIR /home/django/backend
RUN useradd -m -s /bin/bash django
RUN apt update && apt install -y netcat-openbsd && \
    apt clean all

COPY --chown=django . /home/django/backend
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
RUN mkdir /home/django/venv && \
    chown -R django:django /home/django/venv

USER django
RUN  python3 -m venv /home/django/venv && \
     pip3 install -r requirements.txt --no-cache-dir
EXPOSE 8000
ENTRYPOINT ["/entrypoint.sh"]

FROM base as dev
CMD python3 manage.py runserver 0.0.0.0:8000

FROM base AS prod
CMD python3 -m gunicorn backend.wsgi:application --bind 0.0.0.0:8000
