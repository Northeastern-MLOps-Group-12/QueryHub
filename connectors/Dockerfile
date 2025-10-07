FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev python3-dev musl-dev \
 && rm -rf /var/lib/apt/lists/*

COPY connectors/requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy everything (since you build from root)
COPY connectors/ connectors/
COPY databases/ databases/
COPY .env .

ENV PYTHONPATH=/app
EXPOSE 8080

CMD ["python", "connectors/main.py"]
