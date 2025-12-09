"""
LangGraph Workflow - COMPLETE with SQL Complexity Analysis + Workflow Error Tracking
"""

from dotenv import load_dotenv
import warnings
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from . import generate_sql_query
from .state import AgentState
from . import sql_runner
from .visualization.visualization_generator import generate_visualizations
from .query_result_saver import save_query_result
from .database_selector import compute_database_embeddings, select_best_database
from .guardrails import validate_sql_query
from .sql_complexity_analyzer import analyze_sql_complexity
import uuid
from typing import Tuple, Any, Dict

# ✅ MONITORING IMPORT (NEW)
from backend.monitoring import track_workflow_error

warnings.filterwarnings("ignore")
load_dotenv()


# ============================================================================
# WORKFLOW NODE WRAPPERS WITH ERROR TRACKING (NEW)
# ============================================================================

def compute_db_embeddings_with_tracking(state: AgentState) -> Dict:
    """Compute database embeddings with error tracking"""
    try:
        return compute_database_embeddings(state)
    except Exception as e:
        print(f"❌ Error in database embeddings: {e}")
        track_workflow_error(
            error_stage="database_embeddings",
            error_type=type(e).__name__
        )
        raise


def select_database_with_tracking(state: AgentState) -> Dict:
    """Select database with error tracking"""
    try:
        return select_best_database(state)
    except Exception as e:
        print(f"❌ Error in database selection: {e}")
        track_workflow_error(
            error_stage="database_selection",
            error_type=type(e).__name__
        )
        raise


def get_initial_details_with_tracking(state: AgentState) -> Dict:
    """Get initial details with error tracking"""
    try:
        return generate_sql_query.get_initial_details(state)
    except Exception as e:
        print(f"❌ Error in get_initial_details: {e}")
        track_workflow_error(
            error_stage="get_initial_details",
            error_type=type(e).__name__
        )
        raise


def get_initial_necessary_tables_with_tracking(state: AgentState) -> Dict:
    """Get initial necessary tables with error tracking"""
    try:
        return generate_sql_query.get_inital_necessary_tables(state)
    except Exception as e:
        print(f"❌ Error in get_initial_necessary_tables: {e}")
        track_workflow_error(
            error_stage="get_initial_tables",
            error_type=type(e).__name__
        )
        raise


def rephrase_user_query_with_tracking(state: AgentState) -> Dict:
    """Rephrase user query with error tracking"""
    try:
        return generate_sql_query.rephrase_user_query(state)
    except Exception as e:
        print(f"❌ Error in rephrase_user_query: {e}")
        track_workflow_error(
            error_stage="query_rephrasing",
            error_type=type(e).__name__
        )
        raise


def get_final_required_tables_with_tracking(state: AgentState) -> Dict:
    """Get final required tables with error tracking"""
    try:
        return generate_sql_query.get_final_required_tables(state)
    except Exception as e:
        print(f"❌ Error in get_final_required_tables: {e}")
        track_workflow_error(
            error_stage="get_final_tables",
            error_type=type(e).__name__
        )
        raise


def generate_sql_with_tracking(state: AgentState) -> Dict:
    """Generate SQL with error tracking"""
    try:
        return generate_sql_query.generate_sql_from_tables(state)
    except Exception as e:
        print(f"❌ Error in generate_sql: {e}")
        track_workflow_error(
            error_stage="sql_generation",
            error_type=type(e).__name__
        )
        raise


def analyze_sql_complexity_with_tracking(state: AgentState) -> Dict:
    """Analyze SQL complexity with error tracking"""
    try:
        return analyze_sql_complexity(state)
    except Exception as e:
        print(f"❌ Error in analyze_sql_complexity: {e}")
        track_workflow_error(
            error_stage="sql_complexity_analysis",
            error_type=type(e).__name__
        )
        raise


def validate_sql_with_tracking(state: AgentState) -> Dict:
    """Validate SQL with error tracking"""
    try:
        return validate_sql_query(state)
    except Exception as e:
        print(f"❌ Error in validate_sql: {e}")
        track_workflow_error(
            error_stage="sql_validation",
            error_type=type(e).__name__
        )
        raise


def execute_sql_with_tracking(state: AgentState) -> Dict:
    """Execute SQL with error tracking"""
    try:
        return sql_runner.execute_sql_query(state)
    except Exception as e:
        print(f"❌ Error in execute_sql: {e}")
        track_workflow_error(
            error_stage="sql_execution",
            error_type=type(e).__name__
        )
        raise


def generate_visualizations_with_tracking(state: AgentState) -> Dict:
    """Generate visualizations with error tracking"""
    try:
        return generate_visualizations(state)
    except Exception as e:
        print(f"❌ Error in generate_visualizations: {e}")
        track_workflow_error(
            error_stage="visualization_generation",
            error_type=type(e).__name__
        )
        raise


def save_query_result_with_tracking(state: AgentState) -> Dict:
    """Save query result with error tracking"""
    try:
        return save_query_result(state)
    except Exception as e:
        print(f"❌ Error in save_query_result: {e}")
        track_workflow_error(
            error_stage="result_saving",
            error_type=type(e).__name__
        )
        raise


# ============================================================================
# WORKFLOW INITIALIZATION
# ============================================================================

def initialize_agent() -> Tuple[Any, MemorySaver, str]:
    """
    Initialize and compile the LangGraph workflow agent - COMPLETE
    Called ONCE on server startup.
    """
    print("Building LangGraph workflow...")
    
    memory = MemorySaver()
    workflow = StateGraph(AgentState)

    # ========================================================================
    # ADD NODES WITH ERROR TRACKING (UPDATED)
    # ========================================================================
    
    # Database selection nodes
    workflow.add_node("compute_db_embeddings", compute_db_embeddings_with_tracking)
    workflow.add_node("select_database", select_database_with_tracking)

    # SQL generation nodes
    workflow.add_node("get_initial_details", get_initial_details_with_tracking)
    workflow.add_node("get_inital_necessary_tables", get_initial_necessary_tables_with_tracking)
    workflow.add_node("rephrase_user_query", rephrase_user_query_with_tracking)
    workflow.add_node("get_final_required_tables", get_final_required_tables_with_tracking)
    workflow.add_node("generate_sql", generate_sql_with_tracking)
    
    # SQL complexity analysis node
    workflow.add_node("analyze_sql_complexity", analyze_sql_complexity_with_tracking)
    
    # Validation and execution nodes
    workflow.add_node("validate_sql", validate_sql_with_tracking)
    workflow.add_node("execute_sql", execute_sql_with_tracking)
    workflow.add_node("generate_visualizations", generate_visualizations_with_tracking)
    workflow.add_node("save_query_result", save_query_result_with_tracking)

    # ========================================================================
    # EDGES - Define workflow flow
    # ========================================================================
    
    # Start with database selection
    workflow.add_edge(START, "compute_db_embeddings")
    
    # Check for errors after embeddings
    def check_error_after_embeddings(state: AgentState) -> str:
        if state.error:
            return "end"
        return "continue"
    
    workflow.add_conditional_edges(
        "compute_db_embeddings",
        check_error_after_embeddings,
        {
            "continue": "select_database",
            "end": END
        }
    )
    
    # Check for errors after selection
    def check_error_after_selection(state: AgentState) -> str:
        if state.error:
            return "end"
        return "continue"
    
    workflow.add_conditional_edges(
        "select_database",
        check_error_after_selection,
        {
            "continue": "get_initial_details",
            "end": END
        }
    )

    # SQL generation flow
    workflow.add_edge("get_initial_details", "get_inital_necessary_tables")
    workflow.add_edge("get_inital_necessary_tables", "rephrase_user_query")
    workflow.add_edge("rephrase_user_query", "get_final_required_tables")
    workflow.add_edge("get_final_required_tables", "generate_sql")
    
    # Add complexity analysis after SQL generation
    workflow.add_edge("generate_sql", "analyze_sql_complexity")
    
    # Validate SQL after complexity analysis
    workflow.add_edge("analyze_sql_complexity", "validate_sql")
    
    # Check validation result
    def check_validation_result(state: AgentState) -> str:
        if state.error:
            return "end"
        return "execute"
    
    workflow.add_conditional_edges(
        "validate_sql",
        check_validation_result,
        {
            "execute": "execute_sql",
            "end": END
        }
    )

    # Retry logic after execution
    def should_retry_sql_generation(state: AgentState) -> str:
        if state.execution_success:
            return "visualize"
        
        if len(state.prev_errors) < state.max_retries:
            return "retry"
        
        return "end"

    workflow.add_conditional_edges(
        "execute_sql",
        should_retry_sql_generation,
        {
            "retry": "generate_sql",  # Retry from SQL generation
            "visualize": "generate_visualizations",
            "end": END
        }
    )

    workflow.add_edge("generate_visualizations", "save_query_result")
    workflow.add_edge("save_query_result", END)

    # Compile the workflow
    agent = workflow.compile(checkpointer=memory)
    
    # Generate global session ID
    global_session_id = str(uuid.uuid4())[:8]
    
    print("✓ Workflow nodes added")
    print("✓ Workflow edges configured")
    print("✓ SQL complexity analysis enabled")
    print("✓ Workflow error tracking enabled")
    print("✓ Retry attempt tracking enabled")
    print("✓ Agent compiled with memory checkpointing")
    
    return agent, memory, global_session_id


