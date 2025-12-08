from connectors.connector import Connector
from vectorstore.chroma_vector_store import ChromaVectorStore
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os
from langsmith import traceable
from langsmith.run_helpers import trace
from backend.utils.vectorstore_gcs import upload_vectorstore_to_gcs

# Get the embedding model name from environment variables
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


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

    # Upload to GCS after building
    try:
        print("Uploading vector store to GCS...")
        print(f"Local vector store path: {vector_store.persist_directory}")
        print(f"User ID: {config['user_id']}, DB Name: {config['db_name']}")
        
        upload_vectorstore_to_gcs(
            local_vectorstore_path=vector_store.persist_directory,  # Changed from local_vectorstore_path
            user_id=config['user_id'],
            db_name=config['db_name']
        )
        print(f"✅ Vector store uploaded to GCS for {config['db_name']}")
    except Exception as e:
        print(f"⚠️ Warning: Failed to upload vector store to GCS: {e}")
        # Don't fail the entire operation if GCS upload fails
        # The vector store is still available locally

    return state