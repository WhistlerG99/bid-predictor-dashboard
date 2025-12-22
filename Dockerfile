FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    redis-server \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --disable-pip-version-check -r requirements.txt

COPY . .

EXPOSE 8000

CMD redis-server --daemonize yes && python dash_app.py