FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    redis-server \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --default-timeout=100 --disable-pip-version-check -r requirements.txt

ARG BID_PREDICTOR_REPO=https://github.com/WhistlerG99/bid-predictor.git
ARG BID_PREDICTOR_REF=snapshot-data
RUN if [ -n "$BID_PREDICTOR_REPO" ]; then \
        git clone --depth 1 --branch "$BID_PREDICTOR_REF" "$BID_PREDICTOR_REPO" /opt/bid_predictor \
        && pip install --no-cache-dir --disable-pip-version-check -e /opt/bid_predictor; \
    fi

COPY . .

EXPOSE 8000

CMD redis-server --daemonize yes && python dash_app.py