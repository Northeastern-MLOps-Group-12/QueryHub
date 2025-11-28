FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime

# Install your other dependencies
RUN apt-get update && apt-get install -y git curl && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your scripts
COPY train.py /app/train.py
COPY model_eval.py /app/model_eval.py
COPY experiment_utils.py /app/experiment_utils.py

WORKDIR /app

CMD ["python3", "train.py"]