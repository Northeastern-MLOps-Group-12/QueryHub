FROM python:3.10-slim

WORKDIR /app

# Copy only the requirements first (for better caching)
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Copy only the application code, not the entire repository
COPY api/connectors_api.py .

EXPOSE 8080
CMD ["uvicorn", "connectors_api:app", "--host", "0.0.0.0", "--port", "8080"]