"""
LangGraph Workflow - COMPLETE with SQL Complexity Analysis
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
from .sql_complexity_analyzer import analyze_sql_complexity  # NEW
import uuid
from typing import Tuple, Any

warnings.filterwarnings("ignore")
load_dotenv()


def initialize_agent() -> Tuple[Any, MemorySaver, str]:
    """
    Initialize and compile the LangGraph workflow agent - COMPLETE
    Called ONCE on server startup.
    """
    print("Building LangGraph workflow...")
    
    memory = MemorySaver()
    workflow = StateGraph(AgentState)

    # Add database selection nodes (FIRST)
    workflow.add_node("compute_db_embeddings", compute_database_embeddings)
    workflow.add_node("select_database", select_best_database)

    # Add SQL generation nodes
    workflow.add_node("get_initial_details", generate_sql_query.get_initial_details)
    workflow.add_node("get_inital_necessary_tables", generate_sql_query.get_inital_necessary_tables)
    workflow.add_node("rephrase_user_query", generate_sql_query.rephrase_user_query)
    workflow.add_node("get_final_required_tables", generate_sql_query.get_final_required_tables)
    workflow.add_node("generate_sql", generate_sql_query.generate_sql_from_tables)
    
    # Add SQL complexity analysis node (NEW)
    workflow.add_node("analyze_sql_complexity", analyze_sql_complexity)
    
    # Add validation and execution nodes
    workflow.add_node("validate_sql", validate_sql_query)
    workflow.add_node("execute_sql", sql_runner.execute_sql_query)
    workflow.add_node("generate_visualizations", generate_visualizations)
    workflow.add_node("save_query_result", save_query_result)

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
    
    # NEW: Add complexity analysis after SQL generation
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
    print("✓ Error handling enabled")
    print("✓ Agent compiled with memory checkpointing")
    
    return agent, memory, global_session_id