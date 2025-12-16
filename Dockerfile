FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# App Runner will connect to this port
EXPOSE 8000

# No gunicorn – just run the script directly
CMD ["python", "dash_app.py"]
