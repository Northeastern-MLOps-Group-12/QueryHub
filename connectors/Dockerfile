FROM python:3.10-slim

WORKDIR /app

# Install system dependencies needed for psycopg2
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libpq-dev \
    python3-dev \
    musl-dev \
 && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI code
COPY api/ api/

# Expose the default Cloud Run port (optional, good practice)
EXPOSE 8080

# Use a main.py entrypoint to read PORT dynamically
COPY main.py .

# Run the app
CMD ["python", "main.py"]
