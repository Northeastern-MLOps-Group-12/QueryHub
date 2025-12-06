#!/bin/bash
set -e

# Authenticate with GCP if service account key exists
if [ -f "/opt/airflow/keys/service-account-key.json" ]; then
    echo "Configuring Docker authentication for Artifact Registry..."
    
    # Set gcloud config directory
    export CLOUDSDK_CONFIG=/tmp/gcloud-config
    mkdir -p $CLOUDSDK_CONFIG
    
    # Activate service account
    gcloud auth activate-service-account \
        --key-file=/opt/airflow/keys/service-account-key.json \
        --quiet
    
    # Configure docker credential helper
    gcloud auth configure-docker us-east1-docker.pkg.dev --quiet
    
    # Copy the updated docker config to airflow's home (where Python docker client looks)
    mkdir -p /home/airflow/.docker
    cp ~/.docker/config.json /home/airflow/.docker/config.json 2>/dev/null || true
    
    echo "Docker authentication configured successfully"
fi

# Execute the original Airflow command
exec /entrypoint "$@"