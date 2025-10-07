FROM python:3.10-slim

WORKDIR /app

# Install system dependencies required for psycopg2 and other packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY api/connectors_api.py .

EXPOSE 8080
CMD ["uvicorn", "connectors_api:app", "--host", "0.0.0.0", "--port", "8080"]
