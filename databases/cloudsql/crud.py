from sqlalchemy.orm import Session
from .models.credentials import Credentials
from databases.cloudsql.database import get_db

def create_record(db: Session, model_class, data: dict):
    """
    Generic function to create a record in any table.

    Args:
        db (Session): SQLAlchemy session.
        model_class: SQLAlchemy ORM model class (e.g., Credentials, Users).
        data (dict): Dictionary of field values for the record.

    Returns:
        The newly created record.
    """
    new_record = model_class(**data)
    db.add(new_record)
    db.commit()
    db.refresh(new_record)
    return new_record

def delete_record(db: Session, model_class, user_id: int, db_name: str):
    """
    Delete a record from any table by user_id and db_name.

    Args:
        db (Session): SQLAlchemy session.
        model_class: SQLAlchemy ORM model class (e.g., Credentials).
        user_id (str): The user's ID.
        db_name (str): The database name.

    Returns:
        bool: True if record was deleted, False if not found.
    """
    record = db.query(model_class).filter(
        model_class.user_id == user_id,
        model_class.db_name == db_name
    ).first()
    
    if record:
        db.delete(record)
        db.commit()
        return True
    return False


def update_record(db: Session, model_class, user_id: int, connection_name: str, data: dict):
    """
    Update a record in any table by user_id and connection_name.

    Args:
        db (Session): SQLAlchemy session.
        model_class: SQLAlchemy ORM model class (e.g., Credentials).
        user_id (int): The user's ID.
        connection_name (str): The connection name.
        data (dict): Dictionary of fields to update.

    Returns:
        The updated record if found, None otherwise.
    """
    record = db.query(model_class).filter(
        model_class.user_id == int(user_id),
        model_class.connection_name == connection_name
    ).first()
    
    if record:
        for key, value in data.items():
            if hasattr(record, key):
                setattr(record, key, value)
        db.commit()
        db.refresh(record)
        return record
    return None