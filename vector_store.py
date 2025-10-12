import os
from langchain_openai.embeddings import OpenAIEmbeddings
from dotenv import load_dotenv
from langchain.vectorstores import Chroma
import json
from generate_table_data import generate_data
from utils.summarizers.dataset_summarizer import generate_dataset_description

load_dotenv()

DIR_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),"VectorStores")
embedding_function = OpenAIEmbeddings(model="text-embedding-3-large")


def does_vectorstore_exists(db_name):
    if os.path.exists(os.path.join(DIR_PATH,f"chroma_{db_name}")):
        return True
    return False

def form_vector_store(db_name , db_conn_str):
    if does_vectorstore_exists(db_name):
        print("Vector Store Already Exist")
        return
    
    persist_directory = os.path.join(DIR_PATH,f"chroma_{db_name}")

    vector_store = Chroma(
        collection_name=f"chroma_{db_name}_schema",
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

    tables_metadata, table_descriptions , dataset_desc = generate_data(conn_str = db_conn_str)
    
    # print(db_conn_str)

    # print(tables_metadata,table_descriptions,dataset_desc)

    for table_meta, table_desc in zip(tables_metadata, table_descriptions):
        vector_store.add_texts(
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
    
    vector_store.add_texts(
        texts=[f"**Dataset Details**:\n\n{dataset_desc}"],
        metadatas=[{
            "Dataset Summary":dataset_desc
        }],
        ids = [f"{db_name} Summary"]
    )

    vector_store.persist()
    print("Vector store created and saved to:", persist_directory)

def search_vector_store(db_name, query, top_k):
    persist_directory = os.path.join(DIR_PATH, f"chroma_{db_name}")
    collection_name = f"chroma_{db_name}_schema"

    if not does_vectorstore_exists(db_name):
        print(f"No vector store found for {db_name}. Build it first!")
        return

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

    # Perform similarity search (get more than top_k to allow filtering)
    candidate_docs = vector_store.similarity_search(query, k=top_k)  

    # Filter only table-level docs (IDs starting with 'table' or 'Table')
    table_docs = [
        doc for doc in candidate_docs 
        if len(doc.metadata.keys())>1
    ]

    # Return only top_k filtered docs
    return table_docs[:top_k]

def get_vector_store(db_name):
    persist_directory = os.path.join(DIR_PATH, f"chroma_{db_name}")
    collection_name = f"chroma_{db_name}_schema"

    vector_store = Chroma(
        collection_name=collection_name,
        embedding_function=embedding_function,
        persist_directory=persist_directory
    )

    return vector_store