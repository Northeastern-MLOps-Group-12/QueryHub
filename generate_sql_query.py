from global_state import AgentState
import vector_store
from langchain_openai import ChatOpenAI
import json

llm = ChatOpenAI(model="gpt-4.1-nano")

def restructure_docs_with_seperate_keys(docs):
    restructured_docs = []

    for meta, text in zip(docs["metadatas"], docs["documents"]):
        if len(meta.keys()) > 1:  # table-level metadata
            restructured_docs.append({
                "table_name": meta["table_name"],
                "primary_key": meta["primary_key"],
                "foreign_keys": meta["foreign_keys"],
                "columns": meta["columns"],
                "indexes": meta["indexes"],
                "description": text 
            })
    
    return restructured_docs

def restructure_docs(docs):
    restructured_docs = []

    for doc in docs:
        meta = doc.metadata
        if len(meta.keys()) > 1:  
            restructured_docs.append({
                "table_name": meta["table_name"],
                "primary_key": meta["primary_key"],
                "foreign_keys": meta["foreign_keys"],
                "columns": meta["columns"],
                "indexes": meta["indexes"],
                "description": doc.page_content.split("\n\n")[1] 
            })
    
    return restructured_docs

def get_initial_details(state:AgentState):
    if state.dataset_description!="":
        return {
            "dataset_description":state.dataset_description,
            "all_table_details":state.all_table_details
        }
    else:
        vector_db = vector_store.get_vector_store(db_name=state.db_name)

        docs = vector_db.get()

        desc = [doc for doc in docs["metadatas"] if len(doc.keys())==1][0]["Dataset Summary"]
        
        restructured_all_tables = restructure_docs_with_seperate_keys(docs=docs)

        return {
            "dataset_description": desc,
            "all_table_details": restructured_all_tables
        }

def get_inital_necessary_tables(state:AgentState):

    docs = vector_store.search_vector_store(db_name=state.db_name , query = state.user_query, top_k = state.initial_top_k)

    print(docs ,state.initial_top_k )

    restructured_inital_necesssary_tables = restructure_docs(docs=docs)

    return {
        "initial_necessary_table_details":restructured_inital_necesssary_tables
    }

def rephrase_user_query(state: AgentState):
    """
    Rephrases the user's natural language query into a technical, database-aware form
    and estimates how many tables are needed to answer it.
    Returns a dict in the format: {"rephrased_query": "", "top_k": int}
    """

    user_query = state.user_query
    dataset_description = state.dataset_description
    tables_descriptions = [doc["description"] for doc in state.initial_necessary_table_details]

    # Format table descriptions clearly
    formatted_tables = "\n\n".join(
        [f"Table {i+1}:\n{desc}" for i, desc in enumerate(tables_descriptions)]
    )

    prompt = f"""
    You are an expert data analyst with deep knowledge of relational databases, SQL, 
    and database schema design.

    Your task is to take a user's natural-language query and convert it into a 
    precise, technical query that clearly specifies:

    1. Which tables are likely needed.
    2. Which columns from each table are relevant.
    3. Any relationships or joins (foreign keys) that might be required.
    4. Important details from the dataset description that could affect the query.

    You are given the following information:

    **Dataset Description:**  
    {dataset_description}

    **Table Details:**  
    {formatted_tables}

    **User Query:**  
    {user_query}

    ---

    ### Instructions:
    - Rephrase the user's query into a single, concise, technical sentence suitable for database reasoning.
    - Consider table relevance carefully. Include only tables that are likely necessary to answer the query.
    - Identify the key columns needed from each table. You may mention them implicitly in your technical query.
    - If multiple tables are involved, hint at how they might join (via foreign keys, etc.).
    - Estimate the approximate number of tables needed (`top_k`) to answer this query.
    - Do not generate SQL statements.
    - Do not include lists, markdown, or extra explanations.
    - Output strictly **as a JSON object** in this exact format:

    {{
    "rephrased_query": "<single-sentence technical query>",
    "top_k": <integer number of tables likely needed>
    }}

    No extra text, no markdown, only the JSON object.
    """

    response = llm.invoke(prompt)
    raw_output = response.content.strip()

    try:
        result = json.loads(raw_output)
        if not isinstance(result, dict):
            raise ValueError
    except Exception:
        result = {"rephrased_query": raw_output, "top_k": 4}

    if "top_k" in result:
        try:
            result["top_k"] = int(result["top_k"])
        except Exception:
            result["top_k"] = 3

    return {
        "rephrased_query": result["rephrased_query"], "top_k": int(result["top_k"]) + 1
    }

def get_final_required_tables(state:AgentState):
    rephrased_query = state.rephrased_query

    top_k = state.top_k
    
    docs = vector_store.search_vector_store(db_name=state.db_name , query = rephrased_query , top_k=top_k)

    final_necessary_table_details = restructure_docs(docs=docs) 

    return {
        "final_necessary_table_details": final_necessary_table_details
    }