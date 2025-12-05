from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class AgentState(BaseModel):
    # User & Session Info
    user_id: str = Field(default="123", description="User identifier for session management")
    session_id: str = Field(default="", description="Session/thread ID from LangGraph")
    
    # Database & Query Info
    db_config : Dict = Field(default={}, description="Database configuration")
    db_name: str = Field(default="", description="The company DB name")
    user_query: str = Field(default="", description="The Natural Language query to be converted to SQL")
    rephrased_query: str = Field(default="", description="Rephrased user query containing more technical and data terms")
    
    # Table Details
    dataset_description: str = Field(default="", description="The summary of the database")
    initial_necessary_table_details: List = Field(default=[], description="List of relevant Tables Necessary for Making SQL query")
    all_table_details: List = Field(default=[], description="List of all Tables in the dataset")
    final_necessary_table_details: List = Field(default=[], description="Final List of relevant Tables Necessary for Making SQL query")
    
    # Vector Store Config
    initial_top_k: int = Field(default=10, description="Number of tables to fetch initially from the database before rephrasing")
    top_k: int = Field(default=4, description="Number of tables to fetch from the database after rephrasing")
    
    # SQL Generation
    generated_sql: str = Field(default="", description="Generated SQL Query")
    prev_sqls: List[str] = Field(default=[], description="Previous Incorrect SQL queries")
    prev_errors: List[str] = Field(default=[], description="Previous Incorrect SQL's corresponding errors")
    
    # SQL Execution
    query_results: List[Dict] = Field(default=[], description="Results from SQL query execution as list of dicts")
    execution_success: bool = Field(default=False, description="Whether SQL execution was successful")
    execution_error: Optional[str] = Field(default=None, description="Error message if SQL execution failed")
    max_retries: int = Field(default=3, description="Maximum number of retry attempts for SQL generation")
    
    # Visualization
    visualization_intent: str = Field(default="bi", description="Intent: 'bi' for business intelligence or 'eda' for exploratory analysis")
    generated_visualizations: List[Dict] = Field(default=[], description="List of visualization specifications")
    visualization_metadata: Dict = Field(default={}, description="Metadata about visualizations for rendering")
    
    # Cloud Storage
    local_viz_path: str = Field(default="", description="Local path where visualizations are saved")
    cloud_viz_files: List[Dict] = Field(default=[], description="List of cloud file URLs and metadata")
    upload_success: bool = Field(default=False, description="Whether cloud upload was successful")

    # Query Result Storage
    result_saved: bool = Field(default=False, description="Whether query result was saved to GCS")
    result_gcs_path: str = Field(default="", description="GCS path to saved parquet file")
    result_url: str = Field(default="", description="URL to access saved parquet file")
    result_metadata_path: str = Field(default="", description="GCS path to metadata JSON")

    database_metadata: Dict = Field(default={}, description="...")
    available_databases: List[str] = Field(default=[], description="...")
    selected_db_similarity: float = Field(default=0.0, description="...")
    database_selection_ranking: List[Dict] = Field(default=[], description="...")
    db_config: Dict = Field(default={}, description="...")