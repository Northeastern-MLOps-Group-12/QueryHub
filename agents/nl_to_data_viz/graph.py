from dotenv import load_dotenv
import warnings
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver
from . import generate_sql_query
from .state import AgentState
from vectorstore.chroma_vector_store import ChromaVectorStore
from .sql_runner import get_conn_str
from . import sql_runner
from .visualization.visualization_generator import generate_visualizations
from .query_result_saver import save_query_result
from .database_selector import compute_database_embeddings, select_best_database
import uuid
import os
import json
from pathlib import Path

warnings.filterwarnings("ignore")
load_dotenv()

def build_visualization(user_query: str, user_id: str):
    memory = MemorySaver()

    # Initialize workflow
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
    workflow.add_node("execute_sql", sql_runner.execute_sql_query)
    workflow.add_node("generate_visualizations", generate_visualizations)
    workflow.add_node("save_query_result", save_query_result)

    # Add edges - NEW: Start with database selection
    workflow.add_edge(START, "compute_db_embeddings")
    workflow.add_edge("compute_db_embeddings", "select_database")
    workflow.add_edge("select_database", "get_initial_details")

    # Existing edges
    workflow.add_edge("get_initial_details", "get_inital_necessary_tables")
    workflow.add_edge("get_inital_necessary_tables", "rephrase_user_query")
    workflow.add_edge("rephrase_user_query", "get_final_required_tables")
    workflow.add_edge("get_final_required_tables", "generate_sql")
    workflow.add_edge("generate_sql", "execute_sql")

    # Conditional edge with retry logic
    def should_retry_sql_generation(state: AgentState) -> str:
        if state.execution_success:
            return "visualize"
        elif len(state.prev_errors) < state.max_retries:
            return "retry"
        return "end"

    workflow.add_conditional_edges(
        "execute_sql",
        should_retry_sql_generation,
        {
            "retry": "generate_sql",
            "visualize": "generate_visualizations",
            "end": END
        }
    )

    workflow.add_edge("generate_visualizations", "save_query_result")
    workflow.add_edge("save_query_result", END)

    agent = workflow.compile(checkpointer=memory)

    session_id = str(uuid.uuid4())[:8]
    

    print(f"Session ID: {session_id}")

    config = {"configurable": {"thread_id": session_id}}

    st = {
        "user_id": user_id,
        "session_id": session_id,
        "user_query": user_query,
        "db_name": "",
        "db_config": {},
        "dataset_description": "",
        "initial_necessary_table_details": [],
        "all_table_details": [],
        "final_necessary_table_details": [],
        "rephrased_query": "",
        "top_k": 4,
        "initial_top_k": 10,
        "generated_sql": "",
        "prev_sqls": [],
        "prev_errors": [],
        "max_retries": 2,
        "local_viz_path": os.path.join(os.getcwd(), "query_visualizations"),
        "database_metadata": {},
        "available_databases": [],
        "selected_db_similarity": 0.0,
        "database_selection_ranking": []
    }

    result1 = agent.invoke(input=st, config=config)

    output_file = Path("result1.json")

    # Save dictionary as JSON
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result1, f, indent=2, ensure_ascii=False)

    print(f"Saved dictionary to {output_file}")

    # print("_________________________________", result1, "_________________________________")

    print(f"\n{'='*60}")
    print(f"QUERY 1 RESULTS")
    print(f"{'='*60}")
    print(f"Selected DB: {result1['db_name']}")
    print(f"Similarity: {result1['selected_db_similarity']:.3f}")
    print(f"Embeddings computed: {result1['database_metadata']['computed']}")
    print(f"\nGenerated SQL:\n{result1['generated_sql']}")
    print(f"\nDashboard URL: {result1['visualization_metadata']['dashboard_url']}")
    print(f"Data URL: {result1['result_url']}")
