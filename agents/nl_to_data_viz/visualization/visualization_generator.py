# ============================================================================
# FILE: AgentFiles/Visualization/visualization_generator.py (FINAL)
# ============================================================================

import pandas as pd
import os
from typing import Dict
from pathlib import Path
from datetime import timedelta
from dotenv import load_dotenv
from ..state import AgentState
from ..utils.serialization_helpers import sanitize_for_serialization
from .column_analyzer import ColumnAnalyzer
from .bi_visualizer import BIVisualizer
from .eda_visualizer import EDAVisualizer
from .viz_renderer import VisualizationRenderer
from .cloud_uploader import GCSUploader
from .dashboard_generator import DashboardGenerator

load_dotenv()

_analyzer = ColumnAnalyzer()
_bi_visualizer = BIVisualizer(_analyzer)
_eda_visualizer = EDAVisualizer(_analyzer)
_renderer = VisualizationRenderer()
_dashboard_generator = DashboardGenerator()

GCS_CONFIG = {
    'enabled': os.getenv('GCS_UPLOAD_ENABLED', 'true').lower() == 'true',
    'project_id': os.getenv('PROJECT_ID'),
    'bucket_name': os.getenv('GCS_BUCKET_NAME'),
    'credentials_path': os.getenv('GCS_CREDENTIALS_PATH'),
    'make_public': os.getenv('GCS_MAKE_PUBLIC', 'true').lower() == 'true',
    'signed_url_hours': int(os.getenv('GCS_SIGNED_URL_HOURS', '24'))
}


def detect_intent(state: AgentState, df: pd.DataFrame) -> str:
    """Multi-signal intent detection with weighted scoring"""
    sql = state.generated_sql.upper()
    query = state.user_query.lower()
    
    bi_score = 0
    eda_score = 0
    
    if "GROUP BY" in sql:
        bi_score += 50
    
    agg_functions = ["SUM(", "AVG(", "COUNT(", "MAX(", "MIN(", "STRING_AGG(", "LISTAGG("]
    if any(func in sql for func in agg_functions):
        bi_score += 40
    
    if "HAVING" in sql:
        bi_score += 30
    
    if "SELECT *" in sql:
        eda_score += 50
    
    select_clause = sql.split('FROM')[0] if 'FROM' in sql else sql
    column_count = select_clause.count(',') + 1
    
    if column_count > 10:
        eda_score += 40
    elif column_count <= 5 and "GROUP BY" in sql:
        bi_score += 20
    
    join_count = sql.count('JOIN')
    if join_count >= 2 and "GROUP BY" not in sql:
        eda_score += 30
    
    if "WHERE" in sql and "GROUP BY" not in sql:
        eda_score += 15
    
    n_rows = len(df)
    n_cols = len(df.columns)
    
    if n_rows == 1:
        bi_score += 20
    
    if n_cols > 10:
        eda_score += 20
    elif n_cols <= 3:
        bi_score += 15
    
    high_uniqueness_cols = sum(
        1 for col in df.columns 
        if df[col].nunique() / len(df) > 0.8
    )
    
    if high_uniqueness_cols >= n_cols * 0.7:
        eda_score += 15
    
    eda_keywords = ["show me", "give me all", "list all", "contents", 
                    "everything", "explore", "see all", "display all", "all"]
    bi_keywords = ["total", "sum", "average", "count", "top", 
                   "by category", "breakdown", "grouped"]
    
    if any(kw in query for kw in eda_keywords):
        eda_score += 10
    
    if any(kw in query for kw in bi_keywords):
        bi_score += 10
    
    if bi_score > eda_score + 20:
        return "bi"
    elif eda_score > bi_score + 20:
        return "eda"
    else:
        if "GROUP BY" in sql or any(func in sql for func in agg_functions):
            return "bi"
        else:
            return "eda"


def generate_visualizations(state: AgentState) -> Dict:
    print("_____________________generate_visualizations______________________")
    """Complete visualization pipeline"""
    df = pd.DataFrame(state.query_results)
    
    if df.empty:
        return {
            "generated_visualizations": [],
            "visualization_metadata": {"error": "No data to visualize"},
            "visualization_intent": "bi",
            "local_viz_path": "",
            "cloud_viz_files": [],
            "upload_success": False
        }
    
    intent = detect_intent(state, df)
    
    if intent == "bi":
        visualizations = _bi_visualizer.generate(
            df, 
            state.generated_sql, 
            state.final_necessary_table_details
        )
    else:
        visualizations = _eda_visualizer.generate(df)
    
    if not visualizations:
        return {
            "visualization_intent": intent,
            "generated_visualizations": [],
            "visualization_metadata": {"error": "No suitable columns for visualization"},
            "local_viz_path": "",
            "cloud_viz_files": [],
            "upload_success": False
        }
    
    session_path = _renderer.create_session_folder(
        user_id=state.user_id,
        db_name=state.db_name,
        session_id=state.session_id
    )
    
    render_result = _renderer.render_all(df, visualizations, session_path)
    
    dashboard_metadata = {
        'intent': intent,
        'total_graphs': len(visualizations),
        'data_rows': len(df),
        'data_columns': len(df.columns),
        'rendered_files': len(render_result['files']),
        'timestamp': render_result['metadata']['timestamp']
    }
    
    local_dashboard_path = _dashboard_generator.generate_dashboard(
        session_path=session_path,
        visualizations=visualizations,
        metadata=dashboard_metadata
    )
    
    cloud_files = []
    upload_success = False
    dashboard_url = None
    error = False
    error_message = ""
    
    if GCS_CONFIG['enabled'] and GCS_CONFIG['project_id'] and GCS_CONFIG['bucket_name']:
        try:
            uploader = GCSUploader(
                project_id=GCS_CONFIG['project_id'],
                credentials_path=GCS_CONFIG.get('credentials_path')
            )
            
            cloud_files = uploader.upload_folder(
                local_path=session_path,
                bucket_name=GCS_CONFIG['bucket_name'],
                user_id=state.user_id,
                make_public=GCS_CONFIG['make_public'],
                signed_url_expiration_hours=GCS_CONFIG['signed_url_hours']
            )
            
            cloud_dashboard_path = session_path / "dashboard_cloud.html"
            _dashboard_generator.generate_cloud_dashboard(
                cloud_files=cloud_files,
                visualizations=visualizations,
                metadata=dashboard_metadata,
                output_path=cloud_dashboard_path
            )
            
            from google.cloud import storage
            client = storage.Client(project=GCS_CONFIG['project_id'])
            bucket = client.bucket(GCS_CONFIG['bucket_name'])
            
            folder_name = session_path.name
            blob_name = f"visualizations/{state.user_id}/{folder_name}/dashboard.html"
            blob = bucket.blob(blob_name)
            blob.upload_from_filename(str(cloud_dashboard_path), content_type='text/html')
            
            if GCS_CONFIG['make_public']:
                public_url = f"https://storage.googleapis.com/{GCS_CONFIG['bucket_name']}/{blob_name}"
                dashboard_url = public_url
            else:
                dashboard_url = blob.generate_signed_url(
                    expiration=timedelta(hours=GCS_CONFIG['signed_url_hours']),
                    method='GET'
                )
            
            upload_success = True
            error = False
            error_message = ""
            
            print(f"Uploaded {len(cloud_files)} files to GCS")
            
        except Exception as e:
            print(f"GCS upload failed: {e}")
            error = True
            error_message = "Failed to upload to GCS. Please try again."
            upload_success = False
            dashboard_url = None
    
    result = {
        "visualization_intent": intent,
        "generated_visualizations": visualizations,
        "visualization_metadata": {
            "intent": intent,
            "total_graphs": len(visualizations),
            "data_rows": len(df),
            "data_columns": len(df.columns),
            "rendered_files": len(render_result['files']),
            "dashboard_url": dashboard_url,
            "dashboard_local_path": str(local_dashboard_path)
        },
        "local_viz_path": str(session_path),
        "cloud_viz_files": cloud_files,
        "upload_success": upload_success,
        "error": error,
        "error_message": error_message
    }
    
    return sanitize_for_serialization(result)