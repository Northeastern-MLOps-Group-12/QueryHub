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

# if __name__ == "__main__":
#     db = next(get_db())

#     # Add a credential
#     credential_data = {
#         "user_id": 435,
#         "name": "sdfs",
#         "provider": "rtrs",
#         "db_type": "dgd",
#         "db_user": "dre",
#         "db_password": "xyz",
#         "dbname": "rtet",
#         "user": "dfg",
#     }
#     credential = create_record(db, Credentials, credential_data)

#     db.close()
