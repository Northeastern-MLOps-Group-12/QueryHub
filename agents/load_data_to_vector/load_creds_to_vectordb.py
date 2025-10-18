from connectors.connector import Connector
from vectorstore.chroma_vector_store import ChromaVectorStore
# from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")

def save_creds_to_gcp(state):
    config = state.creds
    engine = state.engine or "postgres"

    connector = Connector.get_connector(engine=engine, config=config)
    conn = connector.connect()
    connector.analyze_and_save()

    print("✅ Credentials saved to GCP Secret Manager successfully.")
    state.engine = engine
    state.creds = config
    return state

def build_vector_store(state):
    config = state.creds
    engine = state.engine or "postgres"

    connector = Connector.get_connector(engine=engine, config=config)
    # embedding_fn = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL)

    vector_store = ChromaVectorStore(
        db_name=config['db_name'],
        embedding_model=EMBEDDING_MODEL,
        model='gemini'
    )
    vector_store.build(connector=connector)
    print(vector_store.get_store())

    print(f"✅ Vector store built for {config['db_name']}")
    # state.description = f"Vector store for {config['db_name']} built successfully."
    return state