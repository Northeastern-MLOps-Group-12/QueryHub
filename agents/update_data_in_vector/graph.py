from langgraph.graph import StateGraph , END , START
from langgraph.checkpoint.memory import MemorySaver
from .state import AgentState
from .update_creds_in_vectordb import update_creds_in_gcp, build_vector_store


def build_graph_to_update():
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("update_creds", update_creds_in_gcp)
    graph.add_node("build_vector_store", build_vector_store)

    # Define flow
    graph.add_edge(START, "update_creds")
    graph.add_edge("update_creds", "build_vector_store")
    graph.add_edge("build_vector_store", END)

    # Add memory for state checkpoints
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)
