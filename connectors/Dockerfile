FROM python:3.10-slim

WORKDIR /app

# Install all system dependencies needed to compile psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    python3-dev \
    musl-dev \
 && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

COPY api/connectors_api.py .

EXPOSE 8080
CMD ["uvicorn", "connectors_api:app", "--host", "0.0.0.0", "--port", "8080"]
