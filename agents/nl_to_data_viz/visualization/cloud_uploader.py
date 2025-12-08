# ============================================================================
# FILE: AgentFiles/Visualization/cloud_uploader.py
# ============================================================================
"""
Google Cloud Storage uploader for visualizations
"""

import os
from datetime import timedelta
from pathlib import Path
from typing import List, Dict


class GCSUploader:
    """Upload visualizations to Google Cloud Storage"""
    
    def __init__(self, project_id: str = None, credentials_path: str = None):
        """
        Initialize GCS uploader
        
        Args:
            project_id: GCP project ID
            credentials_path: Path to service account JSON key file
        """
        from google.cloud import storage
        
        # if credentials_path and os.path.exists(credentials_path):
        #     os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = credentials_path
        
        # if project_id:
        #     self.client = storage.Client(project=project_id)
        # else:
        #     self.client = storage.Client()
        self.client = storage.Client(project=project_id)
        self.project_id = project_id or self.client.project
    
    def upload_folder(self, local_path: Path, bucket_name: str, 
                     user_id: str, make_public: bool = False,
                     signed_url_expiration_hours: int = 24) -> List[Dict]:
        """
        Upload entire folder to GCS
        
        Returns:
            List of file metadata with URLs
        """
        bucket = self.client.bucket(bucket_name)
        
        if not bucket.exists():
            raise ValueError(f"Bucket '{bucket_name}' does not exist. Create it first.")
        
        uploaded_files = []
        folder_name = local_path.name
        gcs_prefix = f"visualizations/{user_id}/{folder_name}"
        
        for file_path in local_path.glob("*"):
            if not file_path.is_file():
                continue
            
            blob_name = f"{gcs_prefix}/{file_path.name}"
            blob = bucket.blob(blob_name)
            
            # Set content type
            content_type = self._get_content_type(file_path)
            blob.upload_from_filename(str(file_path), content_type=content_type)
            
            # Make public if requested
            if make_public:
                blob.make_public()
            
            file_info = {
                'filename': file_path.name,
                'gcs_path': f"gs://{bucket_name}/{blob_name}",
                'blob_name': blob_name
            }
            
            # Add appropriate URL
            if make_public:
                file_info['url'] = blob.public_url
                file_info['url_type'] = 'public'
            else:
                file_info['url'] = blob.generate_signed_url(
                    expiration=timedelta(hours=signed_url_expiration_hours),
                    method='GET'
                )
                file_info['url_type'] = 'signed'
                file_info['expires_hours'] = signed_url_expiration_hours
            
            uploaded_files.append(file_info)
        
        return uploaded_files
    
    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type from file extension"""
        suffix = file_path.suffix.lower()
        content_types = {
            '.html': 'text/html',
            '.json': 'application/json',
            '.png': 'image/png',
            '.jpg': 'image/jpeg',
            '.svg': 'image/svg+xml'
        }
        return content_types.get(suffix, 'application/octet-stream')
    
    def create_bucket_if_not_exists(self, bucket_name: str, 
                                   location: str = "US") -> bool:
        """Create bucket if it doesn't exist"""
        bucket = self.client.bucket(bucket_name)
        
        if bucket.exists():
            return False
        
        bucket = self.client.create_bucket(bucket_name, location=location)
        print(f"✓ Created bucket: {bucket.name} in {bucket.location}")
        return True
    
    def set_bucket_lifecycle(self, bucket_name: str, days: int = 30):
        """Set lifecycle rule to auto-delete old files"""
        bucket = self.client.bucket(bucket_name)
        bucket.add_lifecycle_delete_rule(age=days)
        bucket.patch()
        print(f"✓ Lifecycle set: delete files older than {days} days")

