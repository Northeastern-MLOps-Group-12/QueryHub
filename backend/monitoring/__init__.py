"""
QueryHub Monitoring Package - COMPLETE

All monitoring features:
- System resources (CPU, Memory, Disk)
- Request rate (RPS)
- LLM token metrics (TTFT, tokens/sec)
- SQL complexity analysis
- Component timing
- Error tracking
"""

from .metrics import (
    # System resource metrics
    system_cpu_usage,
    system_memory_usage,
    system_memory_available,
    system_disk_usage,
    process_cpu_usage,
    process_memory_usage,
    update_system_metrics,
    
    # Request rate metrics
    requests_per_second,
    query_requests_total,
    record_request,
    
    # Latency metrics
    query_processing_duration,
    
    # LLM token metrics
    llm_time_to_first_token,
    llm_total_generation_time,
    llm_tokens_generated,
    llm_tokens_per_second,
    llm_average_time_per_token,
    track_llm_generation,
    track_llm_timing,
    
    # Component timing
    database_selection_duration,
    sql_generation_duration,
    sql_validation_duration,
    sql_execution_duration,
    visualization_generation_duration,
    gcs_upload_duration,
    
    # Error metrics
    sql_validation_failures,
    sql_execution_errors,
    retry_attempts,
    workflow_errors,
    
    # Database metrics
    database_queries,
    database_selection_similarity,
    
    # SQL Complexity metrics
    sql_complexity_distribution,
    sql_complexity_score,
    sql_features_detected,
    sql_average_complexity_score,
    sql_join_count,
    sql_subquery_nesting_level,
    
    # Visualization metrics
    visualizations_generated,
    visualization_intent,
    charts_per_query,
    gcs_upload_status,
    
    # System health
    active_sessions,
    agent_initialization_status,
    system_uptime_seconds,
    query_results_rows,
    
    # Tracking functions
    track_query_request,
    track_retry_count,
    track_database_selection,
    track_visualization_intent,
    track_charts_generated,
    track_query_result_size,
    track_workflow_error,
    track_sql_error,
    track_validation_failure,
    track_sql_complexity,
    get_complexity_summary,
    
    # Utilities
    time_block
)

__all__ = [
    # System metrics
    'system_cpu_usage',
    'system_memory_usage',
    'system_memory_available',
    'system_disk_usage',
    'process_cpu_usage',
    'process_memory_usage',
    'update_system_metrics',
    
    # Request metrics
    'requests_per_second',
    'query_requests_total',
    'record_request',
    'query_processing_duration',
    
    # LLM metrics
    'llm_time_to_first_token',
    'llm_total_generation_time',
    'llm_tokens_generated',
    'llm_tokens_per_second',
    'llm_average_time_per_token',
    'track_llm_generation',
    'track_llm_timing',
    
    # Component timing
    'database_selection_duration',
    'sql_generation_duration',
    'sql_validation_duration',
    'sql_execution_duration',
    'visualization_generation_duration',
    'gcs_upload_duration',
    
    # Error metrics
    'sql_validation_failures',
    'sql_execution_errors',
    'retry_attempts',
    'workflow_errors',
    
    # Database metrics
    'database_queries',
    'database_selection_similarity',
    
    # SQL Complexity metrics
    'sql_complexity_distribution',
    'sql_complexity_score',
    'sql_features_detected',
    'sql_average_complexity_score',
    'sql_join_count',
    'sql_subquery_nesting_level',
    
    # Visualization metrics
    'visualizations_generated',
    'visualization_intent',
    'charts_per_query',
    'gcs_upload_status',
    
    # System health
    'active_sessions',
    'agent_initialization_status',
    'system_uptime_seconds',
    'query_results_rows',
    
    # Functions
    'track_query_request',
    'track_retry_count',
    'track_database_selection',
    'track_visualization_intent',
    'track_charts_generated',
    'track_query_result_size',
    'track_workflow_error',
    'track_sql_error',
    'track_validation_failure',
    'track_sql_complexity',
    'get_complexity_summary',
    'time_block'
]