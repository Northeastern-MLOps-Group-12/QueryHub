from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore")
import os
import vector_store
from connector import get_conn_str
import json

load_dotenv()

def create_database(db_name):
    db_conn_str = get_conn_str(db_name=db_name)
    if vector_store.does_vectorstore_exists(db_name=db_name):
        return "Vector Store Already Exists !!"
    print("Creating Vector Store ...")
    vector_store.form_vector_store(db_name=db_name,db_conn_str=db_conn_str)
    print("Vector Store Created !!")

