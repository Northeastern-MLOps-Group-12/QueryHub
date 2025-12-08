"""
Utility functions for uploading/downloading vector stores to/from GCS
"""
import os
import shutil
from pathlib import Path
from typing import Optional
from google.cloud import storage
from google.oauth2 import service_account


def get_gcs_bucket():
    """Get GCS bucket instance"""
    project_id = os.getenv("PROJECT_ID")
    bucket_name = os.getenv("GCS_MAIN_BUCKET_NAME")
    credentials_path = os.getenv("GCS_CREDENTIALS_PATH")
    
    if not bucket_name:
        raise ValueError("GCS_MAIN_BUCKET_NAME environment variable not set")
    
    # Try to load credentials from file if path is provided
    credentials = None
    if credentials_path:
        # Resolve path relative to backend folder or use absolute path
        cred_path = Path(credentials_path)
        
        # If relative path, try to resolve from backend folder
        if not cred_path.is_absolute():
            # Get backend directory (parent of utils)
            backend_dir = Path(__file__).parent.parent
            cred_path = backend_dir / credentials_path
        
        if cred_path.exists():
            print(f"🔑 Loading credentials from: {cred_path}")
            credentials = service_account.Credentials.from_service_account_file(
                str(cred_path)
            )
        else:
            print(f"⚠️  Credentials file not found at: {cred_path}")
    
    # Initialize client with credentials if available, otherwise use ADC
    if credentials:
        if project_id:
            client = storage.Client(credentials=credentials, project=project_id)
        else:
            client = storage.Client(credentials=credentials)
    else:
        # Fall back to Application Default Credentials
        if project_id:
            client = storage.Client(project=project_id)
        else:
            client = storage.Client()
    
    bucket = client.bucket(bucket_name)
    
    print(f"📦 Using GCS bucket: {bucket_name} (Project: {project_id or 'default'})")
    
    return bucket



def upload_vectorstore_to_gcs(local_vectorstore_path: str, user_id: str, db_name: str) -> str:
    """
    Upload vector store directory to GCS
    
    Args:
        local_vectorstore_path: Local path to the vector store directory
        user_id: User ID
        db_name: Database name
        
    Returns:
        GCS path prefix (e.g., "vectorstores/user_123/db_name")
    """
    try:
        bucket = get_gcs_bucket()
        local_path = Path(local_vectorstore_path)
        print(f"Uploading vector store from {local_path} to GCS bucket {bucket.name}...")
        
        if not local_path.exists():
            raise ValueError(f"Vector store path does not exist: {local_vectorstore_path}")
        
        print(f"Uploading vector store for user_id={user_id}, db_name={db_name}...")
        
        # GCS path prefix: vectorstores/{user_id}/{db_name}/
        gcs_prefix = f"vectorstores/{user_id}/{db_name}"
        print(f"GCS prefix: {gcs_prefix}")
        
        # Upload all files recursively
        uploaded_count = 0
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = Path(root) / file
                # Get relative path from vectorstore directory
                relative_path = local_file_path.relative_to(local_path)
                # GCS blob name
                blob_name = f"{gcs_prefix}/{relative_path}"
                
                try:
                    blob = bucket.blob(blob_name)
                    blob.upload_from_filename(str(local_file_path))
                    uploaded_count += 1
                except Exception as e:
                    # More specific error handling
                    error_msg = str(e)
                    if "does not exist" in error_msg.lower() or "not found" in error_msg.lower():
                        raise ValueError(
                            f"Bucket '{bucket.name}' not found or not accessible. "
                            f"Check: 1) Bucket name is correct, 2) Project ID is correct, "
                            f"3) Service account has Storage permissions"
                        ) from e
                    raise
        
        print(f"✅ Uploaded {uploaded_count} files to gs://{bucket.name}/{gcs_prefix}")
        return f"gs://{bucket.name}/{gcs_prefix}"

    except Exception as e:
            # Re-raise with more context
            raise Exception(
                f"Failed to upload vector store to GCS: {str(e)}. "
                f"Bucket: {os.getenv('GCS_MAIN_BUCKET_NAME')}, Project: {os.getenv('PROJECT_ID')}"
            ) from e


def download_vectorstore_from_gcs(user_id: str, db_name: str, local_vectorstore_path: str) -> bool:
    """
    Download vector store directory from GCS to local path
    """
    try:
        bucket = get_gcs_bucket()
        gcs_prefix = f"vectorstores/{user_id}/{db_name}"
        local_path = Path(local_vectorstore_path)
        
        # List all blobs with this prefix
        blobs = bucket.list_blobs(prefix=gcs_prefix)
        blob_list = list(blobs)
        
        if not blob_list:
            print(f"⚠️ No vector store found in GCS at {gcs_prefix}")
            return False
        
        # Create local directory if it doesn't exist
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Download all files
        downloaded_count = 0
        for blob in blob_list:
            # Get relative path from prefix
            relative_path = blob.name[len(gcs_prefix) + 1:]  # +1 to skip the trailing /
            local_file_path = local_path / relative_path
            
            # Create parent directories if needed
            local_file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download file
            blob.download_to_filename(str(local_file_path))
            
            # IMPORTANT: Set write permissions on downloaded files
            # Files from GCS might be read-only, causing SQLite errors
            os.chmod(local_file_path, 0o644)
            
            downloaded_count += 1
        
        # Set directory permissions too
        os.chmod(local_path, 0o755)
        
        print(f"✅ Downloaded {downloaded_count} files from gs://{bucket.name}/{gcs_prefix} to {local_path}")
        return True
    except Exception as e:
        print(f"⚠️ Failed to download vector store from GCS: {e}")
        return False


def vectorstore_exists_in_gcs(user_id: str, db_name: str) -> bool:
    """
    Check if vector store exists in GCS
    
    Args:
        user_id: User ID
        db_name: Database name
        
    Returns:
        True if vector store exists in GCS
    """
    try:
        bucket = get_gcs_bucket()
        gcs_prefix = f"vectorstores/{user_id}/{db_name}"
        blobs = bucket.list_blobs(prefix=gcs_prefix, max_results=1)
        return len(list(blobs)) > 0
    except Exception as e:
        print(f"Error checking vector store in GCS: {e}")
        return False


def delete_vectorstore_from_gcs(user_id: str, db_name: str) -> bool:
    """
    Delete vector store from GCS
    
    Args:
        user_id: User ID
        db_name: Database name
        
    Returns:
        True if deletion successful
    """
    try:
        bucket = get_gcs_bucket()
        gcs_prefix = f"vectorstores/{user_id}/{db_name}"
        blobs = bucket.list_blobs(prefix=gcs_prefix)
        
        deleted_count = 0
        for blob in blobs:
            blob.delete()
            deleted_count += 1
        
        print(f"✅ Deleted {deleted_count} files from gs://{bucket.name}/{gcs_prefix}")
        return True
    except Exception as e:
        print(f"Error deleting vector store from GCS: {e}")
        return False

def replace_vectorstore_in_gcs(local_vectorstore_path: str, user_id: str, db_name: str) -> str:
    """
    Replace vector store in GCS by deleting old files first, then uploading new ones.
    This ensures only one version exists at a time.
    
    Args:
        local_vectorstore_path: Local path to the vector store directory
        user_id: User ID
        db_name: Database name
        
    Returns:
        GCS path prefix (e.g., "gs://bucket/vectorstores/user_123/db_name")
    """
    try:
        # First, delete old files from GCS if they exist
        print(f"🔄 Replacing vector store in GCS for user_id={user_id}, db_name={db_name}...")
        delete_vectorstore_from_gcs(user_id=user_id, db_name=db_name)
        
        # Then upload the new vector store
        return upload_vectorstore_to_gcs(
            local_vectorstore_path=local_vectorstore_path,
            user_id=user_id,
            db_name=db_name
        )
    except Exception as e:
        raise Exception(
            f"Failed to replace vector store in GCS: {str(e)}"
        ) from e