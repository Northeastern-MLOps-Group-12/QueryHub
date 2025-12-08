# ============================================================================
# FILE: AgentFiles/QueryResults/query_result_saver.py (UPDATED)
# ============================================================================

import pandas as pd
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict
from google.cloud import storage
from .state import AgentState
import os
from dotenv import load_dotenv
import shutil
import time

load_dotenv()


class QueryResultSaver:
    """Save query results and metadata to GCS as parquet"""
    
    def __init__(self):
        self.project_id = os.getenv('PROJECT_ID')
        self.bucket_name = os.getenv('GCS_BUCKET_NAME')
        # self.credentials_path = os.getenv('GCS_CREDENTIALS_PATH')
        
        # if self.credentials_path and os.path.exists(self.credentials_path):
        #     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.credentials_path
        
        # if self.project_id:
        #     self.client = storage.Client(project=self.project_id)
        # else:
        #     self.client = storage.Client()

        self.client = storage.Client(project=self.project_id)
    
    def save_query_result(self, state: AgentState) -> Dict:
        """
        Save query results to GCS parquet file with metadata
        
        Partition: query_results/{user_id}/{timestamp}/
        """
        
        if not state.query_results:
            return {
                "result_saved": False,
                "result_gcs_path": "",
                "result_url": ""
            }
        
        try:
            df = pd.DataFrame(state.query_results)
            timestamp = str(int(time.time()))
            
            partition_path = f"query_results/{state.user_id}/{timestamp}"
            
            temp_dir = Path(f"temp_results/{state.user_id}/{timestamp}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            parquet_filename = "data.parquet"
            metadata_filename = "metadata.json"
            
            local_parquet_path = temp_dir / parquet_filename
            local_metadata_path = temp_dir / metadata_filename
            
            df.to_parquet(str(local_parquet_path), index=False, engine='pyarrow')
            
            metadata = {
                "user_query": state.user_query,
                "generated_sql": state.generated_sql,
                "rephrased_query": state.rephrased_query,
                "db_name": state.db_name,
                "user_id": state.user_id,
                "session_id": state.session_id,
                "timestamp": timestamp,
                "timestamp_readable": datetime.now().isoformat(),
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "dashboard_url": state.visualization_metadata.get('dashboard_url', ''),
                "visualization_intent": state.visualization_intent,
                "execution_success": state.execution_success
            }
            
            with open(local_metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            bucket = self.client.bucket(self.bucket_name)
            
            parquet_blob_name = f"{partition_path}/{parquet_filename}"
            metadata_blob_name = f"{partition_path}/{metadata_filename}"
            
            parquet_blob = bucket.blob(parquet_blob_name)
            parquet_blob.upload_from_filename(str(local_parquet_path), content_type='application/octet-stream')
            
            metadata_blob = bucket.blob(metadata_blob_name)
            metadata_blob.upload_from_filename(str(local_metadata_path), content_type='application/json')
            
            make_public = os.getenv('GCS_MAKE_PUBLIC', 'true').lower() == 'true'
            
            if make_public:
                result_url = f"https://storage.googleapis.com/{self.bucket_name}/{parquet_blob_name}"
                metadata_url = f"https://storage.googleapis.com/{self.bucket_name}/{metadata_blob_name}"
            else:
                signed_hours = int(os.getenv('GCS_SIGNED_URL_HOURS', '24'))
                result_url = parquet_blob.generate_signed_url(
                    expiration=timedelta(hours=signed_hours),
                    method='GET'
                )
                metadata_url = metadata_blob.generate_signed_url(
                    expiration=timedelta(hours=signed_hours),
                    method='GET'
                )
            
            gcs_path = f"gs://{self.bucket_name}/{parquet_blob_name}"
            
            shutil.rmtree(temp_dir.parent.parent, ignore_errors=True)
            
            print(f"Saved query result to: {gcs_path}")
            
            return {
                "result_saved": True,
                "result_gcs_path": gcs_path,
                "result_url": result_url,
                "result_metadata_path": f"gs://{self.bucket_name}/{metadata_blob_name}",
                "result_metadata_url": metadata_url
            }
            
        except Exception as e:
            print(f"Failed to save query result: {e}")
            return {
                "error": True,
                "error_message": f"Failed to save query results.",
                "result_saved": False,
                "result_gcs_path": "",
                "result_url": "",
                "save_error": str(e)
            }


_result_saver = QueryResultSaver()


def save_query_result(state: AgentState) -> Dict:
    print("_____________________save_query_result______________________")
    """Workflow node to save query results to GCS"""
    return _result_saver.save_query_result(state)