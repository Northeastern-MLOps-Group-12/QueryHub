from langchain_openai.chat_models import ChatOpenAI
from dotenv import load_dotenv
from langchain.schema.messages import HumanMessage

load_dotenv()

llm = ChatOpenAI(model = "gpt-4.1-nano")

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

def generate_description(table_metadata):
    """
    Generates a natural-language summary of a SQL table based on its metadata.
    """
    formatted_prompt = prompt.format(table_metadata=table_metadata)
    response = llm([HumanMessage(content=formatted_prompt)])
    return response.content
