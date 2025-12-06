# Use a lightweight Python 3.10 image
FROM python:3.10-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies needed for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev python3-dev musl-dev \
 && rm -rf /var/lib/apt/lists/*  # Clean up apt cache to reduce image size

# Copy Python dependencies file
COPY backend/requirements.txt .

# Upgrade pip and install Python dependencies from requirements.txt
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copy project source code into the container
COPY agents/ agents/
COPY backend/ backend/
COPY connectors/ connectors/
COPY databases/ databases/
COPY vectorstore/ vectorstore/

# Set PYTHONPATH so Python can find all modules in /app
ENV PYTHONPATH=/app

# Expose port 8080 for the app
EXPOSE 8080

# Default command to run the application
# CMD ["python", "backend/main.py"]
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8080"]

