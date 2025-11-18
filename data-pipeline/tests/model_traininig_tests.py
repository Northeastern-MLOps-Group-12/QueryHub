import sys
from unittest.mock import MagicMock, Mock, patch, mock_open
import pytest
import pandas as pd
from airflow.models import DagBag
from pathlib import Path

# Mock ALL Airflow modules before imports
sys.modules['airflow'] = MagicMock()
sys.modules['airflow.models'] = MagicMock()
sys.modules['airflow.models.Variable'] = MagicMock()
sys.modules['airflow.settings'] = MagicMock()
sys.modules['airflow.operators'] = MagicMock()
sys.modules['airflow.operators.python'] = MagicMock()
sys.modules['airflow.utils'] = MagicMock()
sys.modules['airflow.utils.email'] = MagicMock()
sys.modules['airflow.providers'] = MagicMock()
sys.modules['airflow.providers.google'] = MagicMock()
sys.modules['airflow.providers.google.cloud'] = MagicMock()
sys.modules['airflow.providers.google.cloud.hooks'] = MagicMock()
sys.modules['airflow.providers.google.cloud.hooks.gcs'] = MagicMock()
sys.modules['airflow.providers.standard'] = MagicMock()
sys.modules['airflow.providers.standard.operators'] = MagicMock()
sys.modules['airflow.providers.standard.operators.python'] = MagicMock()
sys.modules['airflow.sdk'] = MagicMock()
sys.modules['airflow.sdk.execution_time'] = MagicMock()
sys.modules['airflow.sdk.execution_time.task_runner'] = MagicMock()
sys.modules['airflow.sdk.execution_time.callback_runner'] = MagicMock()
sys.modules['airflow.sdk.bases'] = MagicMock()
sys.modules['airflow.sdk.bases.operator'] = MagicMock()

# Mock the specific classes
GCSHook = MagicMock()
Variable = MagicMock()

# Set up the mock structure
sys.modules['airflow.providers.google.cloud.hooks.gcs'].GCSHook = GCSHook
sys.modules['airflow.models'].Variable = Variable

sys.path.insert(0, str(Path(__file__).parent.parent / 'dags'))


def test_fetch_latest_model_task():
    """
    Test that the 'fetch_latest_model' PythonOperator exists
    and is configured with correct python_callable and op_kwargs.
    """

    dag_bag = DagBag()
    dag = dag_bag.get_dag("train_model_and_save_dag")   # Update if needed

    assert dag is not None, "DAG 'train_model_and_save_dag' not found"

    task = dag.get_task("fetch_latest_model")
    assert task is not None, "Task 'fetch_latest_model' not found in DAG"

    # Verify operator type
    assert isinstance(task, PythonOperator), \
        "'fetch_latest_model' should be a PythonOperator"

    # Verify callable function name
    assert task.python_callable.__name__ == "fetch_latest_model_and_image", \
        "python_callable is not fetch_latest_model_and_image"

    # Verify op_kwargs include GCP project + region
    expected_keys = {"project_id", "region"}
    assert expected_keys.issubset(task.op_kwargs.keys()), \
        f"op_kwargs must include {expected_keys}"

    # Optional: check that op_kwargs pull from Airflow Variables
    assert task.op_kwargs["project_id"] == Variable.get("gcp_project")
    assert task.op_kwargs["region"] == Variable.get("gcp_region")