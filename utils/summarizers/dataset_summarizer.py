from langchain_openai.chat_models import ChatOpenAI
from langchain.schema.messages import HumanMessage
from dotenv import load_dotenv
import json

load_dotenv()

llm = ChatOpenAI(model="gpt-4.1-nano")

prompt = """
You are a data analyst who describes database schemas clearly and professionally.

Given the following list of table metadata (in JSON format), write a concise and insightful **dataset-level description**.  
Your response should summarize:
- The overall purpose of the database (what kind of data it manages)
- The key entities (tables) and their relationships
- Any notable structure (e.g., star schema, transactional, reference data)
- The general scale or richness (based on row counts)
- Example use cases or what this database might support

Be clear and concise — write **1–2 paragraphs**.

Now here is the full table data:
{tables_metadata}
"""

def generate_dataset_description(tables_metadata):
    """
    Generates a natural-language summary of the entire dataset
    based on all table metadata provided.
    """
    formatted_prompt = prompt.format(tables_metadata=json.dumps(tables_metadata, indent=2))
    response = llm([HumanMessage(content=formatted_prompt)])
    return response.content.strip()
