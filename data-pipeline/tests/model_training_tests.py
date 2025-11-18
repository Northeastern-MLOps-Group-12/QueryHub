import sys
from unittest.mock import MagicMock, Mock, patch, mock_open, call
import pytest
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Mock Airflow and Google Cloud modules before imports
sys.modules['airflow'] = MagicMock()
sys.modules['airflow.models'] = MagicMock()
sys.modules['airflow.models.Variable'] = MagicMock()
sys.modules['airflow.operators'] = MagicMock()
sys.modules['airflow.operators.python'] = MagicMock()
sys.modules['airflow.operators.empty'] = MagicMock()
sys.modules['airflow.utils'] = MagicMock()
sys.modules['airflow.utils.email'] = MagicMock()
sys.modules['airflow.exceptions'] = MagicMock()
sys.modules['google.cloud'] = MagicMock()
sys.modules['google.cloud.storage'] = MagicMock()
sys.modules['google.cloud.aiplatform'] = MagicMock()
sys.modules['google.cloud.artifactregistry_v1'] = MagicMock()
sys.modules['sqlglot'] = MagicMock()

# Import after mocking
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.empty import EmptyOperator
from airflow.models import Variable
from airflow.exceptions import AirflowException

# Import the DAG creation function
sys.path.insert(0, str(Path(__file__).parent.parent))
from dags.train_model_and_save import create_model_training_dag, fetch_latest_model, train_on_vertex_ai, register_model_in_vertex_ai, launch_evaluation_job, run_bias_detection, run_syntax_validation_task


class TestModelTrainingDAG:
    """Test cases for the Vertex AI Model Training Pipeline DAG"""

    @pytest.fixture
    def dag(self):
        """Fixture to provide the DAG instance"""
        return create_model_training_dag()

    @pytest.fixture
    def mock_variables(self):
        """Mock Airflow Variables"""
        with patch('dags.train_model_and_save.Variable') as mock_var:
            mock_var.get.side_effect = lambda key: {
                "alert_email": "alerts@example.com",
                "gcp_project": "test-project",
                "gcp_region": "us-central1",
                "gcp_train_data_path": "gs://bucket/train_data.csv",
                "gcp_val_data_path": "gs://bucket/val_data.csv",
                "vertex_ai_training_image_uri": "gcr.io/test/image:latest",
                "vertex_ai_train_machine_type": "n1-standard-4",
                "vertex_ai_train_gpu_type": "NVIDIA_TESLA_T4",
                "serving_container_image_uri": "gcr.io/test/serving:latest",
                "gcp_evaluation_output_csv": "gs://bucket/eval_output",
                "gcp_test_data_path": "gs://bucket/test_data.csv",
                "vertex_ai_eval_machine_type": "n1-standard-2",
                "vertex_ai_eval_gpu_type": "NVIDIA_TESLA_T4",
                "gcs_bias_and_syntax_validation_output": "gs://bucket/bias_output",
                "service_account": "test-service-account@test-project.iam.gserviceaccount.com"
            }.get(key)
            yield mock_var

    def test_dag_creation(self, dag):
        """Test DAG is properly created with correct parameters"""
        assert dag.dag_id == 'vertex_ai_model_training_pipeline'
        assert dag.description == 'Retrain model on Vertex AI using latest image + GCS files'
        assert dag.tags == ['vertex_ai', 'model_training', 'text2sql']
        assert dag.max_active_runs == 1
        assert not dag.catchup

    def test_dag_default_args(self, dag):
        """Test DAG default arguments"""
        expected_args = {
            'owner': 'data-engineering',
            'depends_on_past': False,
            'start_date': datetime(2024, 1, 1),
            'email_on_failure': True,
            'email_on_retry': False,
            'retries': 0,
            'retry_delay': timedelta(minutes=5),
            'execution_timeout': timedelta(hours=6),
        }
        
        for key, value in expected_args.items():
            assert dag.default_args[key] == value

    def test_dag_structure(self, dag):
        """Test DAG task structure and dependencies"""
        tasks = dag.tasks
        task_ids = [task.task_id for task in tasks]
        
        expected_tasks = [
            'start_pipeline', 'pre_test_node', 'fetch_latest_model',
            'train_on_vertex_ai', 'upload_model_to_vertex_ai',
            'evaluate_model_on_vertex_ai', 'bias_detection',
            'syntax_validation', 'training_completed', 'training_failed'
        ]
        
        assert set(task_ids) == set(expected_tasks)
        assert len(tasks) == len(expected_tasks)

    def test_task_dependencies(self, dag):
        """Test task dependencies are correctly set"""
        # Test main success path
        start = dag.get_task('start_pipeline')
        pre_test = dag.get_task('pre_test_node')
        fetch_model = dag.get_task('fetch_latest_model')
        train_model = dag.get_task('train_on_vertex_ai')
        upload_model = dag.get_task('upload_model_to_vertex_ai')
        evaluate_model = dag.get_task('evaluate_model_on_vertex_ai')
        bias_detection = dag.get_task('bias_detection')
        syntax_validation = dag.get_task('syntax_validation')
        training_completed = dag.get_task('training_completed')
        
        # Check downstream dependencies
        assert start.downstream_task_ids == {'pre_test_node'}
        assert pre_test.downstream_task_ids == {'fetch_latest_model'}
        assert fetch_model.downstream_task_ids == {'train_on_vertex_ai'}
        assert train_model.downstream_task_ids == {'upload_model_to_vertex_ai'}
        assert upload_model.downstream_task_ids == {'evaluate_model_on_vertex_ai'}
        assert evaluate_model.downstream_task_ids == {'bias_detection'}
        assert bias_detection.downstream_task_ids == {'syntax_validation'}
        assert syntax_validation.downstream_task_ids == {'training_completed'}

    def test_failure_handler_dependencies(self, dag):
        """Test that all tasks point to training_failed on failure"""
        training_failed = dag.get_task('training_failed')
        
        # All tasks except start_pipeline and training_completed should point to training_failed
        expected_failure_sources = {
            'pre_test_node', 'fetch_latest_model', 'train_on_vertex_ai',
            'upload_model_to_vertex_ai', 'evaluate_model_on_vertex_ai',
            'bias_detection', 'syntax_validation'
        }
        
        assert training_failed.upstream_task_ids == expected_failure_sources

    def test_task_configurations(self, dag, mock_variables):
        """Test individual task configurations"""
        # Test fetch_latest_model task
        fetch_task = dag.get_task('fetch_latest_model')
        assert isinstance(fetch_task, PythonOperator)
        assert fetch_task.python_callable.__name__ == 'fetch_latest_model'
        assert fetch_task.op_kwargs == {
            "project_id": "test-project",
            "region": "us-central1"
        }

        # Test train_on_vertex_ai task
        train_task = dag.get_task('train_on_vertex_ai')
        assert isinstance(train_task, PythonOperator)
        assert train_task.python_callable.__name__ == 'train_on_vertex_ai'
        expected_train_kwargs = {
            "project_id": "test-project",
            "region": "us-central1",
            "gcs_train_data": "gs://bucket/train_data.csv",
            "gcs_val_data": "gs://bucket/val_data.csv",
            "container_image_uri": "gcr.io/test/image:latest",
            "machine_type": "n1-standard-4",
            "gpu_type": "NVIDIA_TESLA_T4",
        }
        assert train_task.op_kwargs == expected_train_kwargs

    @patch('dags.train_model_and_save.storage.Client')
    def test_fetch_latest_model_success(self, mock_storage_client, mock_variables):
        """Test successful execution of fetch_latest_model"""
        # Mock GCS client and blob responses
        mock_bucket = MagicMock()
        mock_blob1 = MagicMock()
        mock_blob1.name = "registered_models/model_v1/weights.bin"
        mock_blob1.time_created = datetime(2024, 1, 1, 10, 0, 0)
        
        mock_blob2 = MagicMock()
        mock_blob2.name = "registered_models/model_v2/weights.bin"
        mock_blob2.time_created = datetime(2024, 1, 2, 10, 0, 0)
        
        mock_bucket.list_blobs.return_value = [mock_blob1, mock_blob2]
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        
        # Mock task instance
        mock_ti = MagicMock()
        
        # Execute function
        result = fetch_latest_model("test-project", "us-central1", ti=mock_ti)
        
        # Verify results
        expected_path = "gs://train_data_query_hub/registered_models/model_v2"
        assert result == expected_path
        mock_ti.xcom_push.assert_called_with(key="latest_model_dir", value=expected_path)

    @patch('dags.train_model_and_save.storage.Client')
    def test_fetch_latest_model_no_models_found(self, mock_storage_client, mock_variables):
        """Test fetch_latest_model when no models are found"""
        mock_bucket = MagicMock()
        mock_bucket.list_blobs.return_value = []  # No blobs found
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        
        mock_ti = MagicMock()
        
        # Should raise AirflowException when no models found
        with pytest.raises(AirflowException, match="No merged models found"):
            fetch_latest_model("test-project", "us-central1", ti=mock_ti)

    @patch('dags.train_model_and_save.aiplatform')
    @patch('dags.train_model_and_save.start_experiment_run')
    @patch('dags.train_model_and_save.log_experiment_params')
    @patch('dags.train_model_and_save.submit_vertex_training_job')
    def test_train_on_vertex_ai_success(self, mock_submit_job, mock_log_params, 
                                      mock_start_experiment, mock_aiplatform, mock_variables):
        """Test successful execution of train_on_vertex_ai"""
        # Mock dependencies
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = "gs://bucket/model_v2"
        
        mock_run = MagicMock()
        mock_start_experiment.return_value = mock_run
        
        mock_submit_job.return_value = "gs://bucket/trained_model"
        
        # Execute function
        train_on_vertex_ai(
            project_id="test-project",
            region="us-central1",
            gcs_train_data="gs://bucket/train_data.csv",
            gcs_val_data="gs://bucket/val_data.csv",
            container_image_uri="gcr.io/test/image:latest",
            machine_type="n1-standard-4",
            gpu_type="NVIDIA_TESLA_T4",
            ti=mock_ti
        )
        
        # Verify experiment was started
        mock_start_experiment.assert_called_once()
        
        # Verify parameters were logged
        mock_log_params.assert_called()
        
        # Verify training job was submitted
        mock_submit_job.assert_called_once()
        
        # Verify XCom push
        mock_ti.xcom_push.assert_any_call(key="experiment_run_name", value=mock.ANY)
        mock_ti.xcom_push.assert_any_call(key="trained_model_gcs", value="gs://bucket/trained_model")

    @patch('dags.train_model_and_save.aiplatform')
    @patch('dags.train_model_and_save.get_experiment_run')
    @patch('dags.train_model_and_save.log_experiment_params')
    def test_register_model_in_vertex_ai_success(self, mock_log_params, mock_get_experiment, 
                                               mock_aiplatform, mock_variables):
        """Test successful model registration"""
        # Mock dependencies
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = "run-20240101-120000"
        
        mock_model = MagicMock()
        mock_model.resource_name = "projects/test-project/locations/us-central1/models/model-123"
        mock_aiplatform.Model.upload.return_value = mock_model
        
        mock_run = MagicMock()
        mock_get_experiment.return_value = mock_run
        
        # Execute function
        result = register_model_in_vertex_ai(
            project_id="test-project",
            region="us-central1",
            model_artifact_path="gs://bucket/trained_model",
            serving_container_image_uri="gcr.io/test/serving:latest",
            ti=mock_ti
        )
        
        # Verify model upload
        mock_aiplatform.Model.upload.assert_called_once()
        
        # Verify experiment logging
        mock_log_params.assert_called_with(mock_run, {"vertex_model_resource": mock_model.resource_name})
        
        # Verify XCom push
        mock_ti.xcom_push.assert_called_with(key="registered_model_name", value=mock_model.resource_name)
        
        # Verify return value
        assert result == mock_model.resource_name

    @patch('dags.train_model_and_save.aiplatform')
    @patch('dags.train_model_and_save.build_output_csv_path')
    def test_launch_evaluation_job_success(self, mock_build_path, mock_aiplatform, mock_variables):
        """Test successful evaluation job launch"""
        # Mock dependencies
        mock_ti = MagicMock()
        mock_build_path.return_value = "gs://bucket/eval_output/output-20240101120000.csv"
        
        mock_eval_job = MagicMock()
        mock_aiplatform.CustomJob.return_value = mock_eval_job
        
        # Execute function
        launch_evaluation_job(
            project_id="test-project",
            region="us-central1",
            output_csv="gs://bucket/eval_output",
            model_registry_id="models/model-123",
            run_name="run-20240101-120000",
            test_data_path="gs://bucket/test_data.csv",
            machine_type="n1-standard-2",
            gpu_type="NVIDIA_TESLA_T4",
            ti=mock_ti
        )
        
        # Verify output path building
        mock_build_path.assert_called_with("gs://bucket/eval_output")
        
        # Verify XCom push
        mock_ti.xcom_push.assert_called_with(
            key='evaluation_output_csv', 
            value="gs://bucket/eval_output/output-20240101120000.csv"
        )
        
        # Verify job creation and execution
        mock_aiplatform.CustomJob.assert_called_once()
        mock_eval_job.run.assert_called_once_with(
            service_account="test-service-account@test-project.iam.gserviceaccount.com", 
            sync=True
        )

    @patch('dags.train_model_and_save.storage.Client')
    @patch('dags.train_model_and_save.upload_to_gcs')
    @patch('dags.train_model_and_save.get_experiment_run')
    @patch('dags.train_model_and_save.log_experiment_metrics')
    def test_run_bias_detection_success(self, mock_log_metrics, mock_get_experiment, 
                                      mock_upload_gcs, mock_storage_client, mock_variables):
        """Test successful bias detection execution"""
        # Mock dependencies
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = None  # No existing folder
        
        # Mock GCS download
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Mock experiment run
        mock_run = MagicMock()
        mock_get_experiment.return_value = mock_run
        
        # Execute function
        result = run_bias_detection(
            project_id="test-project",
            region="us-central1",
            run_name="run-20240101-120000",
            gcs_csv_path="gs://bucket/eval_output.csv",
            gcs_output_path="gs://bucket/bias_output",
            ti=mock_ti
        )
        
        # Verify GCS operations
        mock_blob.download_to_filename.assert_called_once()
        mock_upload_gcs.assert_called()
        
        # Verify XCom push for folder name
        mock_ti.xcom_push.assert_called_with(key="bias_and_syntax_validation_folder", value=mock.ANY)
        
        # Verify experiment metrics logging
        mock_log_metrics.assert_called()

    @patch('dags.train_model_and_save.storage.Client')
    @patch('dags.train_model_and_save.upload_to_gcs')
    @patch('dags.train_model_and_save.aiplatform')
    @patch('dags.train_model_and_save.get_experiment_run')
    @patch('dags.train_model_and_save.log_experiment_metrics')
    def test_run_syntax_validation_success(self, mock_log_metrics, mock_get_experiment, 
                                         mock_aiplatform, mock_upload_gcs, mock_storage_client, mock_variables):
        """Test successful syntax validation execution"""
        # Mock dependencies
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = "bias-20240101120000"  # Folder from bias detection
        
        # Mock GCS operations
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Mock experiment run
        mock_run = MagicMock()
        mock_get_experiment.return_value = mock_run
        
        # Execute function
        run_syntax_validation_task(
            project_id="test-project",
            region="us-central1",
            run_name="run-20240101-120000",
            gcs_csv_path="gs://bucket/eval_output.csv",
            gcs_output_path="gs://bucket/bias_output",
            ti=mock_ti
        )
        
        # Verify GCS operations
        mock_blob.download_to_filename.assert_called_once()
        mock_upload_gcs.assert_called_once()
        
        # Verify experiment operations
        mock_log_metrics.assert_called()
        mock_aiplatform.init.assert_called()
        mock_aiplatform.end_run.assert_called_once()

    def test_task_retry_configuration(self, dag):
        """Test task retry configuration"""
        python_tasks = [task for task in dag.tasks if isinstance(task, PythonOperator)]
        
        for task in python_tasks:
            # All Python tasks should inherit DAG-level retry settings
            assert task.retries == 0
            assert task.retry_delay == timedelta(minutes=5)
            assert task.execution_timeout == timedelta(hours=6)

    def test_email_configuration(self, dag, mock_variables):
        """Test email alert configuration"""
        # Check that email is set from Variable
        assert dag.default_args['email'] == 'alerts@example.com'
        assert dag.default_args['email_on_failure'] is True
        assert dag.default_args['email_on_retry'] is False

    @patch('dags.train_model_and_save.aiplatform')
    def test_train_on_vertex_ai_missing_xcom(self, mock_aiplatform, mock_variables):
        """Test train_on_vertex_ai when XCom data is missing"""
        mock_ti = MagicMock()
        mock_ti.xcom_pull.return_value = None  # No model path from previous task
        
        with pytest.raises(AirflowException, match="Missing GCS model path or container image URI"):
            train_on_vertex_ai(
                project_id="test-project",
                region="us-central1",
                gcs_train_data="gs://bucket/train_data.csv",
                gcs_val_data="gs://bucket/val_data.csv",
                container_image_uri="gcr.io/test/image:latest",
                machine_type="n1-standard-4",
                gpu_type="NVIDIA_TESLA_T4",
                ti=mock_ti
            )

    def test_build_output_csv_path(self):
        """Test output CSV path building function"""
        from dags.train_model_and_save import build_output_csv_path
        
        base_folder = "gs://bucket/eval_output"
        result = build_output_csv_path(base_folder)
        
        assert result.startswith("gs://bucket/eval_output/output-")
        assert result.endswith(".csv")
        
        # Test with trailing slash
        base_folder_with_slash = "gs://bucket/eval_output/"
        result_with_slash = build_output_csv_path(base_folder_with_slash)
        assert not result_with_slash.endswith("//")  # No double slash

    @patch('dags.train_model_and_save.pd.read_csv')
    @patch('dags.train_model_and_save.storage.Client')
    def test_detect_bias_function(self, mock_storage_client, mock_read_csv, mock_variables):
        """Test the detect_bias helper function"""
        from dags.train_model_and_save import detect_bias
        
        # Mock DataFrame with evaluation results
        mock_df = pd.DataFrame({
            'sql_complexity': ['simple', 'complex', 'simple', 'medium'],
            'exact_match': [1, 0, 1, 0],
            'f1_score': [0.9, 0.6, 0.8, 0.7]
        })
        mock_read_csv.return_value = mock_df
        
        # Mock GCS client
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Execute function
        per_bucket, complex_dist = detect_bias("gs://bucket/eval_results.csv")
        
        # Verify results
        assert 'simple' in per_bucket.index
        assert 'complex' in per_bucket.index
        assert 'medium' in per_bucket.index
        
        # Verify distribution counts
        assert complex_dist.loc['simple', 'count'] == 2
        assert complex_dist.loc['complex', 'count'] == 1
        assert complex_dist.loc['medium', 'count'] == 1

    @patch('dags.train_model_and_save.parse_one')
    @patch('dags.train_model_and_save.pd.read_csv')
    @patch('dags.train_model_and_save.storage.Client')
    def test_syntax_validation_from_gcs(self, mock_storage_client, mock_read_csv, mock_parse_one, mock_variables):
        """Test the syntax_validation_from_gcs helper function"""
        from dags.train_model_and_save import syntax_validation_from_gcs
        
        # Mock DataFrame with SQL predictions
        mock_df = pd.DataFrame({
            'predicted_sql': ['SELECT * FROM table', 'INVALID SQL', 'SELECT count(*) FROM users'],
            'sql_complexity': ['simple', 'simple', 'medium']
        })
        mock_read_csv.return_value = mock_df
        
        # Mock SQL parsing - first and third SQL valid, second invalid
        def mock_parse_side_effect(sql, dialect):
            if sql == 'INVALID SQL':
                raise Exception("Invalid SQL")
            return True
        mock_parse_one.side_effect = mock_parse_side_effect
        
        # Mock GCS client
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_storage_client.return_value.bucket.return_value = mock_bucket
        mock_bucket.blob.return_value = mock_blob
        
        # Execute function
        syntax_overall, per_complexity_valid = syntax_validation_from_gcs("gs://bucket/eval_results.csv")
        
        # Verify results
        assert syntax_overall == 2/3  # 2 out of 3 valid
        assert 'simple' in per_complexity_valid.index
        assert 'medium' in per_complexity_valid.index

    def test_dag_schedule(self, dag):
        """Test DAG schedule configuration"""
        # This DAG appears to be triggered manually (no schedule_interval set)
        assert dag.schedule_interval is None

if __name__ == "__main__":
    pytest.main([__file__, "-v"])