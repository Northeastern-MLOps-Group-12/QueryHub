# from msilib import text
import os
import json
from sqlalchemy import text
from dotenv import load_dotenv
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Chroma
from connectors.connector import Connector
from agents.base_agent import Agent
from langsmith import traceable
from langsmith.run_helpers import trace
import chromadb
        

load_dotenv()

DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "VectorStores")

LLM_API_KEY=os.environ["LLM_API_KEY"]

class ChromaVectorStore:
    def __init__(self, user_id: str, db_name: str, embedding_model: str = "text-embedding-3-large", model='gpt'):
        self.user_id = user_id
        self.db_name = db_name
        self.persist_directory = os.path.join(DIR_PATH, f"chroma_{db_name}_{user_id}")
        self.collection_name = f"chroma_{db_name}_schema_{user_id}"
        self.embedding_function = OpenAIEmbeddings(model=embedding_model) if model == "gpt" else GoogleGenerativeAIEmbeddings(model=embedding_model, google_api_key=LLM_API_KEY)
        self.vector_store = None
        self.model = model

    def exists(self) -> bool:
        return os.path.exists(self.persist_directory)

    @traceable(name="build_vector_store")
    def build(self, connector: Connector):
        print("Building vector store...")
        if self.exists():
            trace("Vector Store already exists")
            self.reset()

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

        tables_metadata, table_descriptions, dataset_desc = self.generate_data(connector=connector)
        print("tables_metadata:", tables_metadata)
        print("✅table_descriptions:", table_descriptions)
        print("✅dataset_desc:", dataset_desc)

        for table_meta, table_desc in zip(tables_metadata, table_descriptions):
            self.vector_store.add_texts(
                texts=[f"{table_meta['Table']}\n\n{table_desc['Description']}"],
                metadatas=[{
                    "table_name": table_meta["Table"],
                    "columns": json.dumps(table_meta["Columns"]),
                    "primary_key": json.dumps(table_meta["PrimaryKey"]),
                    "foreign_keys": json.dumps(table_meta["ForeignKeys"]),
                    "indexes": json.dumps(table_meta["Indexes"]),
                    "total_rows": table_meta["TotalRows"]
                }],
                ids=[table_meta["Table"]]
            )

        self.vector_store.add_texts(
            texts=[f"**Dataset Details**:\n\n{dataset_desc}"],
            metadatas=[{"Dataset Summary": dataset_desc}],
            ids=[f"{self.db_name} Summary"]
        )

        self.vector_store.persist()
        trace("✅ Vector store created and saved to:", self.persist_directory)

    def load(self):
        """Load the vector store from disk."""
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embedding_function,
            persist_directory=self.persist_directory
        )

    def search(self, query: str, top_k: int = 5):
        if self.vector_store is None:
            self.load()

        candidate_docs = self.vector_store.similarity_search(query, k=top_k)

        # Filter table-level docs
        table_docs = [doc for doc in candidate_docs if len(doc.metadata.keys()) > 1]

        return table_docs[:top_k]

    def get_store(self):
        """Return the Chroma vector store object."""
        if self.vector_store is None:
            self.load()
        return self.vector_store
    
    @traceable(name="generate_data")
    def generate_data(self, connector: Connector):
        conn = connector.connect()
        inspector = connector.get_inspector()

        tables_metadata = []
        tables = inspector.get_table_names()
        trace("✅Fetching table metadata...", tables)

        for table in tables:
            table_info = {"Table": table}

            # Columns
            columns_info = [
                {
                    "name": col["name"],
                    "type": str(col["type"]),
                    "nullable": col["nullable"],
                    "default": col.get("default"),
                    "autoincrement": col.get("autoincrement")
                }
                for col in inspector.get_columns(table)
            ]
            table_info["Columns"] = columns_info

            # Primary Keys
            pk = inspector.get_pk_constraint(table)
            table_info["PrimaryKey"] = pk.get("constrained_columns")

            # Foreign Keys
            fks = inspector.get_foreign_keys(table)
            table_info["ForeignKeys"] = [
                {
                    "columns": fk["constrained_columns"],
                    "referred_table": fk["referred_table"],
                    "referred_columns": fk["referred_columns"]
                }
                for fk in fks
            ]

            # Indexes
            indexes = inspector.get_indexes(table)
            table_info["Indexes"] = [
                {
                    "name": idx["name"],
                    "columns": idx["column_names"],
                    "unique": idx["unique"]
                }
                for idx in indexes
            ]


            count_result = connector.execute_query(text(f'SELECT COUNT(*) AS total_rows FROM "{table}"'))
            row_count = count_result.fetchone()[0]

            # Total Rows
            table_info["TotalRows"] = row_count

            tables_metadata.append(table_info)

        table_descriptions = []
        # print("✅stables_metadata:", tables_metadata)
        for table_meta in tables_metadata:
            table_name = table_meta["Table"]
            description = self.generate_description(table_meta)
            table_descriptions.append({"Table": table_name, "Description": description})

        # with open("table_descriptions.json","w+") as f:
        #     json.dump(table_descriptions,f)

        dataset_desc = self.generate_description(tables_metadata)

        return tables_metadata , table_descriptions , dataset_desc
    
    def generate_description(self, table_metadata):
        agent = Agent(api_key=os.environ.get("LLM_API_KEY"), model=self.model, model_name=os.environ.get("MODEL_NAME"))
        prompt = """
        You are a data analyst who summarizes SQL table structures clearly and concisely.

        Given the following table metadata (in JSON format), write a short but detailed description of the table. 
        The description should include:
        - The purpose or meaning of the table (inferred from its name and columns)
        - A quick overview of what data it stores
        - A mention of each column with its type, whether nullable, and its general role
        - The primary key(s) and what they uniquely identify
        - Any foreign key relationships (and what tables they connect to)
        - Notable indexes or unique constraints
        - The total number of rows, if relevant

        Be objective and precise — 3–6 sentences maximum.

        Example format:
        "**Table: Orders** — This table stores customer order records. It includes fields such as `OrderID` (primary key), `CustomerID` (foreign key to Customers), and `OrderDate`. Each row represents one placed order. The table contains X rows."

        Now here is the table data:
        {table_metadata}
        """

        prompt_placeholders = {
            "table_metadata": table_metadata
        }

        return agent.generate(prompt, prompt_placeholders)
    
    def get_all_vector_stores(self):
        """Get all collections data as JSON."""
        client = chromadb.PersistentClient(path=self.persist_directory)
        collections = client.list_collections()
        
        all_data = {}
        for col in collections:
            collection = client.get_collection(name=col.name)
            all_data[col.name] = collection.get()
        
        return all_data
    
    def reset(self):
        """Cleanly delete the vector store folder."""
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            print("🧹 Vector store reset successfully")
    
    def delete(self):
        """Delete the vector store for this user and database."""
        import shutil
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
            return True
        return False
