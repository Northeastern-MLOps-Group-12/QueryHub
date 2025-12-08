from .state import AgentState
from vectorstore.chroma_vector_store import ChromaVectorStore
from langchain_openai import ChatOpenAI
import json
from typing import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from ..base_agent import Agent
import os

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")
MODEL = os.getenv("MODEL")
MODEL_NAME = os.getenv("MODEL_NAME")
# llm = ChatOpenAI(model="gpt-4.1-nano")

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
    print("_____________________get_initial_details______________________")
    if state.dataset_description!="":
        return {
            "dataset_description":state.dataset_description,
            "all_table_details":state.all_table_details
        }
    else:
        vector_store = ChromaVectorStore(
            user_id=state.user_id, 
            db_name=state.db_name,        
            embedding_model=EMBEDDING_MODEL,  
            model=MODEL                 
        )
        
        vector_db = vector_store.get_store()

        docs = vector_db.get()

        desc = [doc for doc in docs["metadatas"] if len(doc.keys())==1][0]["Dataset Summary"]
        
        restructured_all_tables = restructure_docs_with_seperate_keys(docs=docs)

        return {
            "dataset_description": desc,
            "all_table_details": restructured_all_tables
        }

def get_inital_necessary_tables(state:AgentState):
    print("_____________________get_inital_necessary_tables______________________")

    vector_store = ChromaVectorStore(
        user_id=state.user_id, 
        db_name=state.db_name,        
        embedding_model=EMBEDDING_MODEL,  
        model=MODEL                 
    )

    docs = vector_store.search_vector_store(db_name=state.db_name , query=state.user_query, top_k=state.initial_top_k, user_id=state.user_id)

    restructured_inital_necesssary_tables = restructure_docs(docs=docs)

    return {
        "initial_necessary_table_details":restructured_inital_necesssary_tables
    }


def rephrase_user_query(state: AgentState):
    print("_____________________rephrase_user_query______________________")
    """
    Rephrases the user's natural language query into a technical, database-aware form
    and estimates how many tables are needed to answer it.
    Returns: {"rephrased_query": "", "top_k": int}
    """

    agent = Agent(api_key=os.environ.get("LLM_API_KEY"), model=MODEL, model_name=MODEL_NAME)
    db_type = state.db_config["db_type"]
    user_query = state.user_query
    dataset_description = state.dataset_description
    tables_descriptions = [doc["description"] for doc in state.initial_necessary_table_details]

    # Format table descriptions
    formatted_tables = "\n\n".join(
        [f"Table {i+1}:\n{desc}" for i, desc in enumerate(tables_descriptions)]
    )



    prompt = """
    You are an expert data analyst with deep knowledge of relational databases, SQL, 
    and database schema design.

    Your task is to take a user's natural-language query and convert it into a 
    precise, technical query that clearly specifies:

    1. Which tables are likely needed.
    2. Which columns from each table are relevant.
    3. Any relationships or joins (foreign keys) that might be required.
    4. Important details from the dataset description that could affect the query.
    5. Return the tables having columns related to the query needed to be answered.

    You are given the following information:

    **Database Type:** {db_type}

    **Dataset Description:**  
    {dataset_description}

    **Table Details:**  
    {formatted_tables}

    **User Query:**  
    {user_query}

    ---

    ### Instructions:
    - Rephrase the user's query into a single, concise, technical sentence suitable for database reasoning.
    - Include the database type (as provided) naturally somewhere in the rephrased query.
    - Consider table relevance carefully. Include only tables that are likely necessary to answer the query.
    - Identify the key columns needed from each table. You may mention them implicitly in your technical query.
    - If multiple tables are involved, hint at how they might join (via foreign keys, etc.).
    - Estimate the approximate number of tables needed (`top_k`) to answer this query.
    - Do not generate SQL statements.
    - Do not include lists, markdown, or extra explanations.
    - Output strictly **as a JSON object** in this exact format:

    {{
    "rephrased_query": "<single-sentence technical query including the database type>",
    "top_k": <integer number of tables likely needed>
    }}

    No extra text, no markdown, only the JSON object.
    """

    prompt_placeholders = {
        "db_type": db_type,
        "dataset_description": dataset_description,
        "formatted_tables": formatted_tables,
        "user_query": user_query
    }

    response = agent.generate(prompt, prompt_placeholders)

    raw_output = response.strip()

    try:
        result = json.loads(raw_output)
    except Exception:
        result = {"rephrased_query": raw_output, "top_k": 3}

    # Ensure top_k is int
    try:
        result["top_k"] = int(result.get("top_k", 3))
    except:
        result["top_k"] = 3

    return {
        "rephrased_query": result["rephrased_query"],
        "top_k": result["top_k"] + 1
    }


def get_final_required_tables(state:AgentState):
    print("_____________________get_final_required_tables______________________")
    rephrased_query = state.rephrased_query

    top_k = state.top_k

    vector_store = ChromaVectorStore(
        user_id=state.user_id, 
        db_name=state.db_name,        
        embedding_model=EMBEDDING_MODEL,  
        model=MODEL                 
    )
    
    docs = vector_store.search_vector_store(db_name=state.db_name , query = rephrased_query , top_k=top_k, user_id=state.user_id)

    final_necessary_table_details = restructure_docs(docs=docs) 

    return {
        "final_necessary_table_details": final_necessary_table_details
    }

def format_table_details(final_tables: list) -> str:
    """
    Format table details from your specific data structure.
    Handles JSON strings for columns, primary_key, foreign_keys, and indexes.
    """
    formatted = []
    
    for table_info in final_tables:
        table_name = table_info.get("table_name", "Unknown")
        description = table_info.get("description", "")
        
        # Parse JSON strings
        try:
            columns = json.loads(table_info.get("columns", "[]"))
            primary_key = json.loads(table_info.get("primary_key", "[]"))
            foreign_keys = json.loads(table_info.get("foreign_keys", "[]"))
            indexes = json.loads(table_info.get("indexes", "[]"))
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON for table {table_name}: {e}")
            continue
        
        # Start building table string
        table_str = f"\n{'='*80}\n**Table: {table_name}**\n{'='*80}"
        
        # Add description if available
        if description:
            table_str += f"\n\n{description}\n"
        
        # Add Primary Key
        if primary_key:
            pk_str = ", ".join(primary_key)
            table_str += f"\n**Primary Key:** {pk_str}"
        
        # Add Foreign Keys with relationships
        if foreign_keys:
            table_str += "\n\n**Foreign Keys:**"
            for fk in foreign_keys:
                fk_cols = ", ".join(fk.get("columns", []))
                ref_table = fk.get("referred_table", "")
                ref_cols = ", ".join(fk.get("referred_columns", []))
                table_str += f"\n  - {fk_cols} → {ref_table}({ref_cols})"
        
        # Add Columns with details
        table_str += "\n\n**Columns:**"
        for col in columns:
            col_name = col.get("name", "")
            col_type = col.get("type", "")
            nullable = "NULL" if col.get("nullable", True) else "NOT NULL"
            default = col.get("default", None)
            autoincrement = col.get("autoincrement", False)
            
            col_str = f"\n  - {col_name}"
            col_str += f" ({col_type})"
            col_str += f" [{nullable}]"
            
            if autoincrement:
                col_str += " [AUTO_INCREMENT]"
            
            if default and default != "null":
                col_str += f" [DEFAULT: {default}]"
            
            table_str += col_str
        
        # Add indexes (optional, for reference)
        if indexes:
            table_str += "\n\n**Indexes:**"
            for idx in indexes:
                idx_name = idx.get("name", "")
                idx_cols = ", ".join(idx.get("columns", []))
                unique = "UNIQUE" if idx.get("unique", False) else "NON-UNIQUE"
                table_str += f"\n  - {idx_name} on ({idx_cols}) [{unique}]"
        
        formatted.append(table_str)
    
    return "\n\n".join(formatted)

def format_error_history(prev_sqls: list, prev_errors: list) -> str:
    """Format previous SQL attempts and errors for the prompt"""
    history = []
    
    for i, (sql, error) in enumerate(zip(prev_sqls, prev_errors), 1):
        history.append(f"\n{'='*80}")
        history.append(f"**Attempt {i}:**")
        history.append(f"{'='*80}")
        history.append(f"\n**SQL Query:**\n{sql}")
        history.append(f"\n**Error Message:**\n{error}")
    
    return "\n".join(history)

def clean_sql_query(sql: str) -> str:
    """Remove markdown formatting and extra whitespace from SQL query"""
    # Remove markdown code blocks
    sql = sql.replace("```sql", "").replace("```", "")
    
    # Remove leading/trailing whitespace
    sql = sql.strip()
    
    # Ensure query ends with semicolon
    if not sql.endswith(";"):
        sql += ";"
    
    return sql

def generate_sql_from_tables(state: AgentState) -> AgentState:
    print("_____________________generate_sql_from_tables______________________")
    """
    Generate SQL query from final necessary tables using LLM.
    Provides comprehensive, dialect-aware instructions to minimize errors.
    Handles error correction and intelligently uses SELECT * if user asks for full table.
    """

    # Extract state variables
    agent = Agent(api_key=os.environ.get("LLM_API_KEY"), model=MODEL, model_name=MODEL_NAME)
    user_query = state.user_query
    rephrased_query = state.rephrased_query
    final_tables = state.final_necessary_table_details
    prev_errors = state.prev_errors
    prev_sqls = state.prev_sqls
    db_name = state.db_name
    db_type = state.db_config.get("db_type", "postgres").lower()

    # Format table details
    table_details_str = format_table_details(final_tables)

    # Dialect-specific instructions
    dialect_instr = ""
    if db_type == "postgres":
        dialect_instr = (
            "Use PostgreSQL syntax and always double-quote identifiers (e.g., \"TableName\", \"ColumnName\"). "
            "Use LIMIT to restrict rows. Ensure functions and data types follow Postgres rules."
        )
    elif db_type == "mysql":
        dialect_instr = (
            "Use MySQL syntax and backtick identifiers if needed (e.g., `TableName`, `ColumnName`). "
            "Use LIMIT to restrict rows. Ensure functions and data types follow MySQL rules."
        )
    elif db_type in ["sqlserver", "mssql_local"]:
        dialect_instr = (
            "Use SQL Server syntax and square brackets for identifiers if needed (e.g., [TableName], [ColumnName]). "
            "Use TOP to restrict rows instead of LIMIT. Ensure functions and data types follow SQL Server rules."
        )
    else:
        dialect_instr = "Use standard SQL syntax appropriate for the database."

    # Comprehensive instruction template
    if len(prev_errors) == 0:
        prompt = """
You are an expert SQL query generator with extensive knowledge of {db_type}.

**Database:** {db_name} ({db_type})

**User Query:** {user_query}

**Rephrased Query:** {rephrased_query}

**Available Tables and Schema:**
{table_details}

**Comprehensive Instructions:**
1. Generate a SQL query that accurately answers the user's question.
2. {dialect_instr}
3. If the user explicitly requests "show all", "list everything", "display full table", 
   "return entire table", "show complete data", or equivalent, use SELECT *.
4. Otherwise, select only relevant columns.
5. Use proper JOINs when multiple tables are required, using only the provided foreign keys.
6. Ensure column names and table names match exactly as shown in the schema.
7. Handle aggregations, GROUP BY, ORDER BY, and nested queries correctly if required.
8. Never invent tables or columns that do not exist in the provided schema.
9. Avoid unnecessary repetition of columns; only include what's required unless SELECT * is requested.
10. Return ONLY the SQL query. Do NOT include explanations, markdown, comments, or extra text.
"""
    else:
        # Error correction mode
        prompt = """
You are an expert SQL query generator with extensive knowledge of {db_type}. 
A previous SQL query attempt failed. Correct it.

**Database:** {db_name}

**User Query:** {user_query}

**Rephrased Query:** {rephrased_query}

**Available Tables and Schema:**
{table_details}

**Previous Attempts and Errors:**
{error_history}

**Comprehensive Instructions:**
1. Carefully analyze previous errors to correct the query.
2. {dialect_instr}
3. If the user explicitly requested all columns or full table contents, use SELECT *.
4. Otherwise, select only relevant columns.
5. Use proper JOINs when multiple tables are required, following the provided foreign keys.
6. Ensure all column and table names match exactly the schema.
7. Correctly handle aggregations, GROUP BY, ORDER BY, and nested queries if needed.
8. Never invent tables or columns that do not exist in the schema.
9. Avoid unnecessary repetition of columns unless required.
10. Return ONLY the corrected SQL query. Do NOT include explanations, markdown, comments, or extra text.
"""

    prompt_placeholders = {
        "db_name": db_name,
        "user_query": user_query,
        "rephrased_query": rephrased_query,
        "table_details": table_details_str,
        "error_history": format_error_history(prev_sqls, prev_errors),
        "dialect_instr": dialect_instr,
        "db_type": db_type
    }

    # Call LLM
    response = agent.generate(prompt, prompt_placeholders)
    generated_sql = clean_sql_query(response.strip())

    return {
        "generated_sql": generated_sql
    }