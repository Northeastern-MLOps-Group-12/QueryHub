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
import uuid
from typing import Tuple, Any

warnings.filterwarnings("ignore")
load_dotenv()


def initialize_agent() -> Tuple[Any, MemorySaver, str]:
    """
    Initialize and compile the LangGraph workflow agent.
    This is called ONCE on server startup.
    
    Returns:
        Tuple of (compiled_agent, memory_instance, global_session_id)
    """
    print("Building LangGraph workflow...")
    
    # Create memory saver for checkpointing
    memory = MemorySaver()
    
    # Initialize workflow with AgentState
    workflow = StateGraph(AgentState)

    # Add database selection nodes (NEW - FIRST)
    workflow.add_node("compute_db_embeddings", compute_database_embeddings)
    workflow.add_node("select_database", select_best_database)

    # Add existing nodes
    workflow.add_node("get_initial_details", generate_sql_query.get_initial_details)
    workflow.add_node("get_inital_necessary_tables", generate_sql_query.get_inital_necessary_tables)
    workflow.add_node("rephrase_user_query", generate_sql_query.rephrase_user_query)
    workflow.add_node("get_final_required_tables", generate_sql_query.get_final_required_tables)
    workflow.add_node("generate_sql", generate_sql_query.generate_sql_from_tables)
    workflow.add_node("validate_sql", validate_sql_query)  # NEW: SQL validation guardrail
    workflow.add_node("execute_sql", sql_runner.execute_sql_query)
    workflow.add_node("generate_visualizations", generate_visualizations)
    workflow.add_node("save_query_result", save_query_result)

    # Add edges - NEW: Start with database selection
    workflow.add_edge(START, "compute_db_embeddings")
    
    # Add conditional edge after compute_db_embeddings to check for errors
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
    
    # Add conditional edge after select_database to check for errors
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

    # Existing edges with error checking
    workflow.add_edge("get_initial_details", "get_inital_necessary_tables")
    workflow.add_edge("get_inital_necessary_tables", "rephrase_user_query")
    workflow.add_edge("rephrase_user_query", "get_final_required_tables")
    workflow.add_edge("get_final_required_tables", "generate_sql")
    
    # NEW: Add validation step after SQL generation, before execution
    workflow.add_edge("generate_sql", "validate_sql")
    
    # Add conditional edge after validation to check for errors
    def check_validation_result(state: AgentState) -> str:
        """
        Check if SQL validation passed.
        If error=True, go to END immediately.
        Otherwise, proceed to execute_sql.
        """
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

    # Conditional edge with retry logic AND error checking
    def should_retry_sql_generation(state: AgentState) -> str:
        """
        Determine the next step after SQL execution.
        Priority:
        1. If successful: proceed to visualization
        2. If failed but retries available: retry SQL generation (goes back to generate_sql, which will re-validate)
        3. If failed and no retries left: END
        """
        
        # Check if execution was successful
        if state.execution_success:
            return "visualize"
        
        # Check if retries are available
        if len(state.prev_errors) < state.max_retries:
            return "retry"
        
        # No retries left, end the workflow
        return "end"

    workflow.add_conditional_edges(
        "execute_sql",
        should_retry_sql_generation,
        {
            "retry": "generate_sql",  # Retry from SQL generation (will re-validate)
            "visualize": "generate_visualizations",
            "end": END
        }
    )

    workflow.add_edge("generate_visualizations", "save_query_result")
    workflow.add_edge("save_query_result", END)

    # Compile the workflow with memory checkpointing
    agent = workflow.compile(checkpointer=memory)
    
    # Generate a fixed global session ID
    # This session ID will be used for ALL requests
    global_session_id = str(uuid.uuid4())[:8]
    
    print("✓ Workflow nodes added")
    print("✓ Workflow edges configured")
    print("✓ Error handling enabled")
    print("✓ Agent compiled with memory checkpointing")
    
    return agent, memory, global_session_id
