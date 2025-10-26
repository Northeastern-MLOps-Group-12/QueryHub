from langgraph.graph import StateGraph , END , START
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .load_creds_to_vectordb import save_creds_to_gcp, build_vector_store


def build_graph():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("save_creds", save_creds_to_gcp)
    graph.add_node("build_vector_store", build_vector_store)

    # Define flow
    graph.add_edge(START, "save_creds")
    graph.add_edge("save_creds", "build_vector_store")
    graph.add_edge("build_vector_store", END)

    # Add memory for state checkpoints
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
