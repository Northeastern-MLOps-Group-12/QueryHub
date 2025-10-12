from pydantic import BaseModel , Field
from typing import List,Dict

class AgentState(BaseModel):
    db_name : str = Field(default="",description="The company DB name.")

    user_query : str = Field(default="",description="The Natural Language query to be converted to SQL")
    rephrased_query:str = Field(default="",description="Rephrased user query containing more technical and data terms.")

    dataset_description : str = Field(default="",description="The summary of the database")
    initial_necessary_table_details : List = Field(default=[] , description="List of relevant Tables Necessary for Making SQL query.")
    all_table_details : List = Field(default=[] , description="List of all Tables int the dataset.")
    final_necessary_table_details : List = Field(default=[] , description="Final List of relevant Tables Necessary for Making SQL query.")
    
    initial_top_k : int =  Field(default=10,description="Number of tables to fetch initially from the database before rephrasing.")
    top_k:int = Field(default=4,description="Number of tables to fetch from the database after rephrasing.")
    
    