# database/cloudsql.py
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get("DATABASE_URL")

print(DATABASE_URL)

engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

# Dependency function to provide a DB session
def get_db():
    """
    Yields a SQLAlchemy database session.
    Ensures the session is closed after use.
    Can be used with FastAPI's dependency injection.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
