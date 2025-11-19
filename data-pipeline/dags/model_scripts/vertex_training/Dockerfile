# Base Python image
FROM python:3.11-slim

# Install system packages
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy training script
COPY train.py /app/train.py
COPY model_eval.py /app/model_eval.py
COPY experiment_utils.py /app/experiment_utils.py

# Set working directory
WORKDIR /app

# Default command
CMD ["python3", "train.py"]