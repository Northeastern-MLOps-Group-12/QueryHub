# connection-manager-service/database/models.py

from sqlalchemy import Column, Integer, String
from ..database import Base


class Credentials(Base):
    __tablename__ = 'creds'
    
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=False) # Foreign key to the User model
    name = Column(String, nullable=False) # Connection name
    instance = Column(String, nullable=False) # Instance Identifier
    provider = Column(String, nullable=False) # e.g., "GCP", "AWS"
    db_type = Column(String, nullable=False) # e.g., "PostgreSQL", "MySQL"
    db_user = Column(String, nullable=False) 
    db_password = Column(String, nullable=False)
    db_name = Column(String, nullable=False)
    user = Column(String, nullable=False)