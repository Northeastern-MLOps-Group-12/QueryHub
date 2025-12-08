"""
Database Selector - FIXED with Correct Variable Names
Automatically chooses embedding model based on EMBD_MODEL_PROVIDER environment variable
"""

import os
import json
from pathlib import Path
from typing import Dict, List
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import numpy as np

from databases.cloudsql.database import get_db
from databases.cloudsql.crud import get_records_by_user_id
from .state import AgentState

# ✅ MONITORING IMPORTS (NEW)
from backend.monitoring import track_database_selection


# ============================================================================
# FIXED: Correct variable names matching generate_sql_query.py
# ============================================================================

# Get environment variables
LLM_API_KEY = os.getenv('LLM_API_KEY')
EMBD_MODEL_PROVIDER = os.getenv('EMBD_MODEL_PROVIDER', 'gemini')  # gemini or gpt
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'text-embedding-004')


class DatabaseSelector:
    """Select best database for a query using semantic similarity"""
    
    def __init__(self):
        """
        Initialize DatabaseSelector with appropriate embedding model
        based on EMBD_MODEL_PROVIDER environment variable
        """
        project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
        self.vector_stores_dir = project_root / "vectorstore" / "VectorStores"
        
        # ====================================================================
        # DYNAMIC EMBEDDING MODEL SELECTION
        # ====================================================================
        
        model_lower = EMBD_MODEL_PROVIDER.lower()
        
        if model_lower in ['gpt', 'openai']:
            # Use OpenAI embeddings for GPT models
            self.embedding_function = OpenAIEmbeddings(
                model="text-embedding-3-large",
                openai_api_key=LLM_API_KEY
            )
            self.embedding_provider = "openai"
            print(f"✅ DatabaseSelector using OpenAI embeddings: text-embedding-3-large")
            
        elif model_lower in ['gemini', 'google']:
            # Use Google embeddings for Gemini models
            self.embedding_function = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=LLM_API_KEY
            )
            self.embedding_provider = "google"
            print(f"✅ DatabaseSelector using Google embeddings: {EMBEDDING_MODEL}")
            
        else:
            # Default to Google embeddings
            self.embedding_function = GoogleGenerativeAIEmbeddings(
                model=EMBEDDING_MODEL,
                google_api_key=LLM_API_KEY
            )
            self.embedding_provider = "google"
            print(f"⚠️  Unknown provider '{EMBD_MODEL_PROVIDER}', defaulting to Google embeddings: {EMBEDDING_MODEL}")
    
    def load_db_configs(self, user_id: str) -> Dict:
        """Load database configurations from database"""
        db = next(get_db())
        creds = get_records_by_user_id(db, int(user_id))
        print(f"Loaded {len(creds)} database credentials for user {user_id}")

        config = {"databases": []}

        for cred in creds:
            db_config = {
                "db_name": cred.db_name,
                "db_type": cred.db_type,
                "db_user": cred.db_user,
                "db_password": cred.db_password,
                "db_host": cred.db_host,
                "db_port": cred.db_port if hasattr(cred, 'db_port') else None
            }
            config["databases"].append(db_config)

        return config
    
    def get_all_vector_stores(self, user_id: str) -> List[str]:
        """Get list of all available vector store database names for user"""
        if not self.vector_stores_dir.exists():
            return []
        
        user_id_str = str(user_id)
        db_names = []

        for folder in self.vector_stores_dir.iterdir():
            if not folder.is_dir():
                continue

            name = folder.name

            # Must start with chroma_ and end with _<userid>
            if not name.startswith("chroma_"):
                continue

            if not name.endswith(f"_{user_id_str}"):
                continue

            # Remove "chroma_" prefix and "_userid" suffix
            core = name[len("chroma_"):]
            dbname = core[: -(len(user_id_str) + 1)]
            db_names.append(dbname)
        
        print(f"Found {len(db_names)} vector stores for user {user_id_str}")
        return db_names
    
    def extract_dataset_description(self, db_name: str, user_id: str) -> str:
        """Extract dataset description from vector store"""
        persist_directory = self.vector_stores_dir / f"chroma_{db_name}_{user_id}"
        collection_name = f"chroma_{db_name}_schema_{user_id}"
        print(f"Extracting description for {db_name} from {persist_directory}")
        
        try:
            vector_store = Chroma(
                collection_name=collection_name,
                embedding_function=self.embedding_function,
                persist_directory=str(persist_directory)
            )
            
            docs = vector_store.get()
            
            for meta in docs["metadatas"]:
                if len(meta.keys()) == 1 and "Dataset Summary" in meta:
                    return db_name + " summary - " + meta["Dataset Summary"]
            
            return ""
            
        except Exception as e:
            print(f"Failed to extract description for {db_name}: {e}")
            return ""
    
    def compute_database_embeddings(self, state: AgentState) -> Dict:
        """
        Compute embeddings for all database descriptions ONCE
        Cached in state for reuse
        """
        # If already computed, return existing data (CACHED)
        if state.database_metadata and state.database_metadata.get('computed'):
            print("✅ Using cached database embeddings")
            return {
                "database_metadata": state.database_metadata,
                "available_databases": state.available_databases
            }
        
        print(f"🔄 Computing database embeddings (Provider: {self.embedding_provider})...")
        
        db_configs = self.load_db_configs(state.user_id)
        available_dbs = self.get_all_vector_stores(state.user_id)
        
        database_metadata = []
        
        for db_name in available_dbs:
            # Get description from vector store
            description = self.extract_dataset_description(db_name, user_id=state.user_id)
            
            # Find matching config
            db_config = None
            for config in db_configs['databases']:
                if config['db_name'] == db_name:
                    db_config = config
                    break
            
            if not db_config:
                print(f"⚠️  Warning: No config found for {db_name}, skipping")
                continue
            
            # Compute embedding for description ONCE
            if description:
                print(f"  📊 Computing embedding for: {db_name}")
                embedding = self.embedding_function.embed_query(description)
            else:
                embedding = []
            
            database_metadata.append({
                'db_name': db_name,
                'description': description,
                'embedding': embedding,
                'config': db_config
            })
        
        print(f"✅ Cached embeddings for {len(database_metadata)} databases")
        
        if not database_metadata:
            return {
                "error": True,
                "error_message": "Connect at least one database or check your internet connection!"
            }

        return {
            "database_metadata": {
                'databases': database_metadata,
                'computed': True
            },
            "available_databases": available_dbs
        }
    
    def select_best_database(self, state: AgentState) -> Dict:
        """
        Select best database for user query using semantic similarity
        Query embedding is computed FRESH each time
        """
        # Ensure metadata is computed
        if not state.database_metadata or not state.database_metadata.get('computed'):
            print("⚠️  Database metadata not computed. Computing now...")
            metadata_result = self.compute_database_embeddings(state)
            database_metadata = metadata_result['database_metadata']
        else:
            database_metadata = state.database_metadata
        
        databases = database_metadata['databases']
        
        if not databases:
            return {
                "error": True,
                "error_message": "No databases available for selection"
            }
        
        # Compute FRESH query embedding every time
        print(f"🔍 Computing query embedding for: '{state.user_query}'")
        query_embedding = self.embedding_function.embed_query(state.user_query)
        
        # Compute similarity scores with CACHED DB embeddings
        similarities = []
        
        for db in databases:
            if not db['embedding']:
                continue
            
            # Cosine similarity between FRESH query and CACHED DB description
            similarity = self._cosine_similarity(query_embedding, db['embedding'])
            
            similarities.append({
                'db_name': db['db_name'],
                'similarity': float(similarity),
                'description': db['description'],
                'config': db['config']
            })
        
        # Sort by similarity (highest first)
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        # Select best match
        best_match = similarities[0]
        
        # ✅ TRACK DATABASE SELECTION (NEW)
        track_database_selection(
            db_name=best_match['db_name'],
            user_id=state.user_id,
            similarity=best_match['similarity']
        )
        
        print(f"\n{'='*70}")
        print(f"📊 Database Selection Results")
        print(f"{'='*70}")
        for i, db in enumerate(similarities, 1):
            marker = "✅" if i == 1 else "  "
            print(f"{marker} {i}. {db['db_name']}: {db['similarity']:.3f}")
        print(f"{'='*70}")
        print(f"🎯 Selected: {best_match['db_name']} (similarity: {best_match['similarity']:.3f})")
        print(f"{'='*70}\n")
        
        # Return updated state fields
        return {
            "db_name": best_match['db_name'],
            "db_config": best_match['config'],
            "selected_db_similarity": best_match['similarity'],
            "database_selection_ranking": similarities
        }
    
    def _cosine_similarity(self, vec1: List, vec2: List) -> float:
        """Compute cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)


# ============================================================================
# GLOBAL INSTANCE & WORKFLOW NODES
# ============================================================================

# Instantiate selector once
_db_selector = DatabaseSelector()


def compute_database_embeddings(state: AgentState) -> Dict:
    """
    Workflow node: Compute database embeddings ONCE
    Cached in state for entire session
    """
    print("_____________________compute_database_embeddings______________________")
    return _db_selector.compute_database_embeddings(state)


def select_best_database(state: AgentState) -> Dict:
    """
    Workflow node: Select best database for CURRENT query
    Query embedding computed fresh, compared to cached DB embeddings
    """
    print("_____________________select_best_database______________________")
    return _db_selector.select_best_database(state)
