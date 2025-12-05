from connectors.connector import Connector
from vectorstore.chroma_vector_store import ChromaVectorStore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from langsmith import traceable
from langsmith.run_helpers import trace

# Get the embedding model name from environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")



def update_creds_in_gcp(state):
    """
    Save database credentials to GCP Secret Manager.

    Args:
        state: An object containing `creds` and `engine` attributes.

    Returns:
        Updated state with credentials saved.
    """
    config = state.creds                  # Get credentials from state
    engine = state.engine or "postgres"   # Default to postgres if engine not specified

    # Create a connector object based on the engine and config
    connector = Connector.get_connector(engine=engine, config=config)
    conn = connector.connect()            # Establish connection
    connector.update_creds(config)          # Update credentials securely in GCP

    trace("✅ Credentials updated in GCP successfully")

    # Update state and return
    state.engine = engine
    state.creds = config
    return state




@traceable(name="build_vector_store")
def build_vector_store(state):
    """
    Build a Chroma vector store for a given database.

    Args:
        state: An object containing `creds` and `engine` attributes.

    Returns:
        Updated state after building the vector store.
    """
    config = state.creds                  # Get credentials from state
    engine = state.engine or "postgres"   # Default engine if not specified

    # Create a connector to interact with the database
    connector = Connector.get_connector(engine=engine, config=config)

    # Initialize the Chroma vector store
    vector_store = ChromaVectorStore(
        user_id=config['user_id'], 
        db_name=config['db_name'],        # Name of the database
        embedding_model=EMBEDDING_MODEL,  # Embedding model to use
        model='gemini'                    # Model type
    )

    if vector_store.exists():
        print("⚠️ Existing vector store found. Resetting...")
        vector_store.reset()

    # Build the vector store from the database schema and data
    vector_store.build(connector=connector)
    print(f"✅ Vector store built for {config['db_name']}")

    return state