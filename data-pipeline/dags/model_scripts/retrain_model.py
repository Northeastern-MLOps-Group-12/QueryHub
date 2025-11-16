import os
import re
import subprocess
from airflow.exceptions import AirflowException
from google.cloud import storage
from google.cloud.artifactregistry_v1 import ArtifactRegistryClient
from model_scripts.train_utils import download_from_gcs_if_needed, submit_vertex_training_job
from airflow.models import Variable
from datetime import datetime

def fetch_latest_model_and_image(project_id, region, **kwargs):
    """
    Fetch the latest Artifact Registry Docker image and latest GCS model folder dynamically.
    Push them to XCom for downstream tasks.
    """
    ti = kwargs["ti"]

    # 1️⃣ Fetch latest GCS model folder by creation time
    client = storage.Client(project=project_id)
    bucket_name = "train_data_query_hub"
    prefix = "registered_models/"

    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix))

    model_folders = {}
    for blob in blobs:
        relative_path = blob.name[len(prefix):].strip("/")
        if "/" not in relative_path:
            continue
        folder_name = relative_path.split("/")[0]
        if folder_name not in model_folders:
            model_folders[folder_name] = blob.time_created
        else:
            if blob.time_created > model_folders[folder_name]:
                model_folders[folder_name] = blob.time_created

    if not model_folders:
        raise AirflowException(f"No models found in gs://{bucket_name}/{prefix}")

    latest_model_folder = max(model_folders, key=lambda k: model_folders[k])
    gcs_model_path = f"gs://{bucket_name}/{prefix}{latest_model_folder}"
    print(f"✅ Latest GCS model path: {gcs_model_path}")
    ti.xcom_push(key="gcs_model_dir", value=gcs_model_path)

    # 2️⃣ Fetch latest Docker image from Artifact Registry
    client_ar = ArtifactRegistryClient()
    repo = f"projects/{project_id}/locations/{region}/repositories/register-models"

    try:
        images = client_ar.list_docker_images(parent=repo)
    except Exception as e:
        raise AirflowException(f"Error fetching images from Artifact Registry: {e}")

    if not images:
        raise AirflowException(f"No Docker images found in {repo}")

    # Build full repo URL prefix
    full_repo_prefix = f"us-east1-docker.pkg.dev/queryhub-473901/register-models/model"

    latest_image = None
    latest_ts = 0

    for image in images:
        # Extract actual image name (example: "model")
        image_name_only = image.name.split('/')[-1].split('@')[0]  

        for tag in image.tags:
            match = re.match(r"v(\d{8}_\d+)$", tag)
            if match:
                ts = int(match.group(1).replace("_", ""))
                if ts > latest_ts:
                    latest_ts = ts
                    latest_image = f"{full_repo_prefix}:{tag}"

            elif tag == "latest":
                latest_image = f"{full_repo_prefix}:{tag}"

    if not latest_image:
        raise AirflowException(f"No suitable Docker image found in {repo}")

    print(f"✅ Latest Artifact Registry image: {latest_image}")
    ti.xcom_push(key="container_image_uri", value=latest_image)

    return gcs_model_path, latest_image


def train_on_vertex_ai(project_id, region, **kwargs):
    """
    Submit Vertex AI Custom Training Job using latest model files + image.
    """
    ti = kwargs["ti"]
    gcs_model_dir = ti.xcom_pull(task_ids="fetch_latest_model", key="gcs_model_dir")
    container_image_uri = ti.xcom_pull(task_ids="fetch_latest_model", key="container_image_uri")

    if not gcs_model_dir or not container_image_uri:
        raise AirflowException("Missing GCS model path or container image URI from XCom")

    print(f"Training model from: {gcs_model_dir} using image: {container_image_uri}")

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    gcs_output_dir = f"gs://train_data_query_hub/trained_models/{timestamp}"
    trained_model_path = submit_vertex_training_job(
        project_id=project_id,
        region=region,
        container_image_uri=container_image_uri,
        gcs_model_dir=gcs_model_dir,
        gcs_output_dir=gcs_output_dir
    )

    # Push output path for next task (image build)
    ti.xcom_push(key="trained_model_gcs", value=trained_model_path)


def build_push_docker_image(project_id, region, artifact_registry_repo, **kwargs):
    """
    Build a Docker image containing trained model files and push to Artifact Registry.
    Uses only a timestamp tag for each run.
    """
    ti = kwargs["ti"]
    trained_model_gcs = ti.xcom_pull(task_ids="train_on_vertex_ai", key="trained_model_gcs")
    # trained_model_gcs = ti.xcom_pull(task_ids="train_locally", key="trained_model_local")

    if not trained_model_gcs:
        raise AirflowException("No trained model path found from previous task")

    print(f"Building new image with trained model from: {trained_model_gcs}")

    # Download model locally
    local_model_dir = download_from_gcs_if_needed(trained_model_gcs)
    # local_model_dir = trained_model_gcs

    # Generate dynamic timestamp tag
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    image_tag = f"{artifact_registry_repo}:v{timestamp}"  # unique tag for this run

    print(f"Building Docker image: {image_tag}")

    dockerfile_content = f"""
    FROM --platform=linux/amd64 python:3.11-slim

    # Install OS-level dependencies needed for some Python packages
    RUN apt-get update && apt-get install -y --no-install-recommends \\
            build-essential \\
            gcc \\
            libffi-dev \\
            libssl-dev \\
            python3-dev \\
            curl \\
        && rm -rf /var/lib/apt/lists/*

    WORKDIR /app
    COPY . /app

    # Install Python dependencies directly (no separate requirements.txt)
    RUN python -m pip install --upgrade pip
    RUN pip install --no-cache-dir \\
            torch==2.0.1 \\
            transformers==4.30.2 \\
            simple-ddl-parser==0.30.0
    """

    docker_build_dir = "/tmp/docker_build"
    os.makedirs(docker_build_dir, exist_ok=True)

    # Copy trained model
    subprocess.run(f"cp -r {local_model_dir} {docker_build_dir}/model", shell=True, check=True)
    with open(f"{docker_build_dir}/Dockerfile", "w") as f:
        f.write(dockerfile_content)

    # Build and Push Docker image with timestamp tag
    # Build Docker image for amd64 explicitly
    subprocess.run(
        f"docker build -t {image_tag} {docker_build_dir}",
        shell=True,
        check=True
    )

    # Push the image to the registry
    subprocess.run(
        f"docker push {image_tag}",
        shell=True,
        check=True
    )
    print(f"✅ Docker image built: {image_tag}")
    print(f"✅ Docker image pushed: {image_tag}")

    # Push image tag info to XCom for downstream tasks
    ti.xcom_push(key="docker_image_tag", value=image_tag)

    return image_tag

# def build_push_docker_image(project_id, region, artifact_registry_repo, **kwargs):
#     """
#     Build a Docker image containing trained model files and push to Artifact Registry
#     using Cloud Build API (no docker, no gcloud).
#     """
#     from google.cloud.devtools.cloudbuild_v1 import CloudBuildClient, Build, BuildStep
#     from google.protobuf.duration_pb2 import Duration

#     ti = kwargs["ti"]
#     trained_model_gcs = ti.xcom_pull(task_ids="train_locally", key="trained_model_local")

#     if not trained_model_gcs:
#         raise AirflowException("No trained model path found from previous task")

#     print(f"Building new image with trained model from: {trained_model_gcs}")

#     # Prepare local build directory
#     docker_build_dir = "/tmp/docker_build"
#     os.makedirs(docker_build_dir, exist_ok=True)

#     # Copy model data
#     subprocess.run(f"cp -r {trained_model_gcs} {docker_build_dir}/model", shell=True, check=True)

#     # Write Dockerfile
#     dockerfile_content = """
#     FROM python:3.11-slim

#     RUN apt-get update && apt-get install -y --no-install-recommends \
#             build-essential \
#             gcc \
#             libffi-dev \
#             libssl-dev \
#             python3-dev \
#             curl \
#         && rm -rf /var/lib/apt/lists/*

#     WORKDIR /app
#     COPY . /app

#     RUN python -m pip install --upgrade pip
#     RUN pip install --no-cache-dir \
#             torch==2.0.1 \
#             transformers==4.30.2 \
#             simple-ddl-parser==0.30.0
#     """

#     with open(f"{docker_build_dir}/Dockerfile", "w") as f:
#         f.write(dockerfile_content)

#     # Create tarball for Cloud Build
#     tar_path = "/tmp/source.tar.gz"
#     subprocess.run(f"tar -czf {tar_path} -C /tmp docker_build", shell=True, check=True)

#     # Upload tarball to your build bucket
#     build_bucket = "cloud_build_mlops"
#     storage_client = storage.Client()
#     bucket = storage_client.bucket(build_bucket)
#     blob_name = f"cloudbuild_source/source-{datetime.now().strftime('%Y%m%d%H%M%S')}.tgz"
#     blob = bucket.blob(blob_name)
#     blob.upload_from_filename(tar_path)

#     # Image tag
#     timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
#     image_tag = f"{artifact_registry_repo}:v{timestamp}"

#     print(f"Building image: {image_tag}")

#     # Cloud Build API client
#     cb_client = CloudBuildClient()

#     # Define build steps
#     build = Build(
#         timeout=Duration(seconds=1800),  # 30 minutes
#         steps=[
#             BuildStep(
#                 name="gcr.io/cloud-builders/docker",
#                 args=["build", "-t", image_tag, "."],
#                 dir="docker_build"
#             ),
#             BuildStep(
#                 name="gcr.io/cloud-builders/docker",
#                 args=["push", image_tag]
#             ),
#         ],
#         images=[image_tag],
#         source={
#             "storage_source": {
#                 "bucket": build_bucket,
#                 "object": blob_name
#             }
#         }
#     )

#     # Submit build
#     operation = cb_client.create_build(project_id=project_id, region=region, build=build)
#     result = operation.result()  # wait

#     print(f"✅ Docker image built and pushed to Artifact Registry: {image_tag}")

#     ti.xcom_push(key="docker_image_tag", value=image_tag)
#     return image_tag
