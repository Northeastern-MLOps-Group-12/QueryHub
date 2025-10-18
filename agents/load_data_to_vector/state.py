from pydantic import BaseModel , Field
from typing import List,Dict

class AgentState(BaseModel):
    engine : str = Field(default="",description="The company DB engine.")
    creds : dict = Field(default={},description="The company DB credentials.")
    # table_metadata : List[Dict] = Field(default=[],description="The table metadata of the company DB.")
    # table_descriptions : List[str] = Field(default=[],description="The table descriptions of the company DB.")
    # description: str = Field(default="",description="The description of the company DB.")
    
    