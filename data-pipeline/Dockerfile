# Using Python 3.11
FROM apache/airflow:3.1.0-python3.11

# Switch to root to install dependencies
USER root

# Install Docker CLI and dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        docker.io \
        curl && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install gcloud SDK
RUN curl -sSL https://sdk.cloud.google.com > /tmp/install.sh && \
    bash /tmp/install.sh --disable-prompts --install-dir=/opt && \
    rm /tmp/install.sh

# Add to PATH
ENV PATH="/opt/google-cloud-sdk/bin:${PATH}"

# Add airflow user to existing docker group (group 999 already exists)
RUN usermod -a -G docker airflow

# Copy entrypoint script
COPY scripts/docker-entrypoint.sh /opt/airflow/scripts/docker-entrypoint.sh
RUN chmod +x /opt/airflow/scripts/docker-entrypoint.sh

# Switch back to airflow user
USER airflow

# Copy and install Python requirements
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt