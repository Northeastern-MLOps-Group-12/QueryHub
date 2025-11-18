from airflow.models import Variable

def pre_test_model_training_pipeline():
    """
    Pre-test function to validate prerequisites before running the pipeline.
    For example: check if required Airflow Variables exist.
    """
    required_vars = ["gcp_project", "gcp_region", "artifact_registry_repo", "cloud_build_bucket"]
    for var in required_vars:
        if not Variable.get(var, default_var=None):
            raise ValueError(f"Required Airflow Variable '{var}' is missing")