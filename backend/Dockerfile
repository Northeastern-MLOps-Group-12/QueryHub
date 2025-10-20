FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev python3-dev musl-dev \
 && rm -rf /var/lib/apt/lists/*

COPY backend/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy everything (since you build from root)
COPY agents/ agents/
COPY backend/ backend/
COPY connectors/ connectors/
COPY databases/ databases/
COPY vectorstore/ vectorstore/


ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["python", "backend/main.py"]
# CMD ["uvicorn", "connectors.main:app", "--host", "0.0.0.0", "--port", "8080"]

