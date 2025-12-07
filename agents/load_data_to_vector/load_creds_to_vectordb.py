"""
Load Database Credentials and Build Vector Store - FIXED
Corrected variable names for embedding model provider
"""

from connectors.connector import Connector
from vectorstore.chroma_vector_store import ChromaVectorStore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from langsmith import traceable
from langsmith.run_helpers import trace

# ============================================================================
# FIXED: Correct variable names
# ============================================================================

# Embedding model name (e.g., "text-embedding-004" or "text-embedding-3-large")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-004")

# Embedding model provider (e.g., "gemini" or "gpt")
EMBD_MODEL_PROVIDER = os.getenv("EMBD_MODEL_PROVIDER", "gemini")


@traceable(name="save_creds_to_gcp")
def save_creds_to_gcp(state):
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
    connector.analyze_and_save()          # Save credentials securely to GCP Secret Manager

    trace("✅ Credentials saved to GCP Secret Manager successfully")

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

    # ========================================================================
    # FIXED: Use correct variable names
    # ========================================================================
    vector_store = ChromaVectorStore(
        user_id=config['user_id'], 
        db_name=config['db_name'],
        embedding_model=EMBEDDING_MODEL,  # Model name: text-embedding-004
        model=EMBD_MODEL_PROVIDER  # Provider: gemini or gpt (FIXED!)
    )

    if vector_store.exists():
        print("⚠️ Existing vector store found. Resetting...")
        vector_store.reset()

    # Build the vector store from the database schema and data
    vector_store.build(connector=connector)
    print(f"✅ Vector store built for {config['db_name']}")
    print(f"   Using embedding provider: {EMBD_MODEL_PROVIDER}")
    print(f"   Using embedding model: {EMBEDDING_MODEL}")

    return state