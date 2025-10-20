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


# if __name__ == "__main__":
#     # Example config (replace with actual DB credentials)
#     config = {
#         "user_id": "111",
#         "connection_name": "chinook",
#         "db_host": "34.139.44.104",
#         "provider": "gcp",
#         "db_type": "postgres",
#         "db_user": "postgres",
#         "db_password": "Ved@Chinook123",
#         "db_name": "postgres"
#     }

#     initial_state = AgentState(engine="postgres", creds=config)
#     graph = build_graph()

#     # final_state = graph.invoke(initial_state)
#     final_state = graph.invoke(
#         initial_state,
#         config={"configurable":{"thread_id":1}}
#     )

#     print()

#     # print("\n🚀 Pipeline finished successfully.")
#     # print(f"Final state: {final_state.model_dump_json(indent=2)}")