"""
Chat utility functions (service layer)
Handles business logic for chat operations
"""

from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any, Tuple
from fastapi import HTTPException
import time

from ..models.chat_model import ChatSummary, ChatDetail, Message
from .agent_utils import build_visualization
import os


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format"""
    return datetime.utcnow().isoformat() + "Z"

# Helper function to safely convert to milliseconds
def to_ms(val):
    if isinstance(val, int): return val
    if hasattr(val, 'timestamp'): return int(val.timestamp() * 1000)
    if isinstance(val, str):
        # Handle 'Z' or standard ISO
        val = val.replace('Z', '+00:00')
        try:
            return int(datetime.fromisoformat(val).timestamp() * 1000)
        except ValueError:
            return 0 # or some default
    return 0

# def get_current_timestamp() -> int:
#     """Get current timestamp in milliseconds"""
#     return int(time.time() * 1000) 

# ==================== Chat Operations ====================

def list_user_chats(db, user_id: str) -> List[ChatSummary]:
    """
    List all chats for a user, ordered by most recent
    
    Args:
        db: Firestore client instance
        user_id: User's unique identifier
        
    Returns:
        List of ChatSummary objects
    """
    try:
        from firebase_admin import firestore
        
        chats_ref = db.collection('chats')\
            .where('user_id', '==', user_id)\
            .order_by('updated_at', direction=firestore.Query.DESCENDING)
        
        chats = chats_ref.stream()
        
        chat_list = []
        for chat in chats:
            chat_data = chat.to_dict()
            
            # Convert Firestore timestamps to milliseconds if needed
            created_at = chat_data['created_at']
            updated_at = chat_data['updated_at']
            
            chat_list.append(ChatSummary(
                chat_id=chat_data['chat_id'],
                chat_title=chat_data['chat_title'],
                created_at=to_ms(created_at),
                updated_at=to_ms(updated_at)
            ))
        
        return chat_list
    
    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail=f"Error fetching chats: {str(e)}"
        )


def create_chat(db, user_id: str, chat_title: str) -> Tuple[str, str]:
    """
    Create a new chat for a user
    
    Args:
        db: Firestore client instance
        user_id: User's unique identifier
        chat_title: Title for the new chat
        
    Returns:
        Tuple of (chat_id, created_at timestamp)
    """
    try:
        # Generate unique chat_id
        chat_ref = db.collection('chats').document()
        chat_id = chat_ref.id
        
        current_time = get_current_timestamp()
        
        chat_data = {
            'chat_id': chat_id,
            'user_id': user_id,
            'chat_title': chat_title,
            'created_at': current_time,
            'updated_at': current_time,
            'history': []
        }
        
        chat_ref.set(chat_data)

        return chat_id,current_time,current_time
    
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error creating chat: {str(e)}"
        )


def get_chat_detail(db, chat_id: str, user_id: str) -> ChatDetail:
    """
    Get full conversation history for a chat
    
    Args:
        db: Firestore client instance
        chat_id: Chat's unique identifier
        user_id: User's unique identifier (for ownership verification)
        
    Returns:
        ChatDetail object with full history
        
    Raises:
        HTTPException: 404 if chat not found, 403 if access denied
    """
    try:
        chat_ref = db.collection('chats').document(chat_id)
        chat_doc = chat_ref.get()
        
        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        chat_data = chat_doc.to_dict()
        
        # Verify ownership
        if chat_data['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Convert Firestore timestamps to milliseconds
        chat_data['created_at'] = to_ms(chat_data['created_at'])
        chat_data['updated_at'] = to_ms(chat_data['updated_at'])
        
        # Convert timestamps in history messages
        history = chat_data.get('history', [])
        for msg in history:
            # Convert 'created_at' for each message using your helper
            msg['created_at'] = to_ms(msg.get('created_at'))
            
            # Ensure message_id is a string (to fix previous errors)
            msg['message_id'] = str(msg.get('message_id'))

        # Update the dictionary with the processed history
        chat_data['history'] = history
        
        return ChatDetail(**chat_data)
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching chat: {str(e)}"
        )


def delete_chat(db, chat_id: str, user_id: str) -> Dict[str, str]:
    """
    Delete a chat
    
    Args:
        db: Firestore client instance
        chat_id: Chat's unique identifier
        user_id: User's unique identifier (for ownership verification)
        
    Returns:
        Success message with chat_id
        
    Raises:
        HTTPException: 404 if chat not found, 403 if access denied
    """
    try:
        chat_ref = db.collection('chats').document(chat_id)
        chat_doc = chat_ref.get()
        
        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        chat_data = chat_doc.to_dict()
        
        # Verify ownership
        if chat_data['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        chat_ref.delete()
        
        return {"message": "Chat deleted successfully", "chat_id": chat_id}
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting chat: {str(e)}"
        )


# ==================== Message Operations ====================

def send_message(db, chat_id: str, user_id: str, message_text: str) -> Dict[str, Any]:
    """
    Send a user message and generate bot response
    
    Args:
        db: Firestore client instance
        chat_id: Chat's unique identifier
        user_id: User's unique identifier
        message_text: User's natural language query
        
    Returns:
        Bot's message dict with response
        
    Raises:
        HTTPException: 404 if chat not found, 403 if access denied
    """
    try:
        chat_ref = db.collection('chats').document(chat_id)
        chat_doc = chat_ref.get()
        
        if not chat_doc.exists:
            raise HTTPException(status_code=404, detail="Chat not found")
        
        chat_data = chat_doc.to_dict()
        
        # Verify ownership
        if chat_data['user_id'] != user_id:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Create user message
        user_message = create_user_message(message_text)
        
        # Generate bot response
        bot_message = generate_bot_response(user_id, chat_id, message_text)
        
        # Update chat with both messages
        update_chat_history(chat_ref, chat_data, [user_message, bot_message])

        if 'content' in bot_message:
            # Remove escape characters and newlines from query
            if 'query' in bot_message['content']:
                query = bot_message['content']['query']
                query = query.replace('\\', '').replace('"', '').replace('\\n', ' ').replace('\n', ' ')
                # Remove extra spaces
                bot_message['content']['query'] = ' '.join(query.split())
            
            # Remove GCS paths
            if 'attachment' in bot_message['content'] and 'gcs_storage_path' in bot_message['content']['attachment']:
                del bot_message['content']['attachment']['gcs_storage_path']
            
            if 'visualization' in bot_message['content'] and 'gcs_storage_path' in bot_message['content']['visualization']:
                del bot_message['content']['visualization']['gcs_storage_path']
        
        return bot_message
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error sending message: {str(e)}"
        )


def create_user_message(text: str) -> Dict[str, Any]:
    """Create a user message dict"""
    message_id = int(datetime.utcnow().timestamp() * 1000)
    
    return {
        'message_id': message_id,
        'sender': 'user',
        'created_at': get_current_timestamp(),
        'content': {
            'text': text
        }
    }


def generate_bot_response(user_id: str, chat_id: str, user_query: str) -> Dict[str, Any]:
    """
    Generate bot response with SQL query execution
    
    TODO: Replace this with actual ML pipeline integration:
    1. Load user's database connector from PostgreSQL
    2. Pass user_query to T5 model for SQL generation
    3. Execute SQL on user's database
    4. Export results to CSV and upload to GCS
    5. Generate visualization HTML and upload to GCS
    6. Return actual metadata
    
    Args:
        user_id: User's unique identifier
        chat_id: Chat's unique identifier
        user_query: User's natural language query
        
    Returns:
        Bot message dict with content
    """
    bot_message_id = int(datetime.utcnow().timestamp() * 1000) + 1
    current_time = get_current_timestamp()
    
    result = build_visualization(user_query, user_id)
    sql_query = result.generated_sql  or ''
    attachment_path = result.result_gcs_path  or ''
    cloud_viz_files = result.cloud_viz_files or List[Dict]
    error = result.error or False
    error_message = result.error_message or ""
    
    # Check if attachment is available
    has_attachment = bool(attachment_path and attachment_path.strip())
    
    if has_attachment and attachment_path.startswith("gs://"):
        attachment_path = attachment_path.split("/", 3)[-1]
    
    # Check if visualization is available
    has_visualization = bool(cloud_viz_files and isinstance(cloud_viz_files, list) and len(cloud_viz_files) > 0)
    
    visualization_path = None
    if has_visualization:
        # Extract dashboard.html path
        dashboard_gcs_path = next(
            (file["gcs_path"] for file in cloud_viz_files if file.get("filename") == "dashboard.html"),
            None
        )
        
        if dashboard_gcs_path and dashboard_gcs_path.startswith("gs://"):
            visualization_path = dashboard_gcs_path.split("/", 3)[-1]
        else:
            has_visualization = False  # No dashboard found
    
    # Build bot message
    bot_message = {
        'message_id': str(bot_message_id),
        'sender': 'bot',
        'created_at': current_time,
        'content': {
            'text': user_query,
            'query': sql_query,
            'attachment': {
                'has_attachment': has_attachment
            },
            'visualization': {
                'has_visualization': has_visualization
            }
        },
        'error': error,
        'error_message': error_message
    }
    
    # Only add gcs_storage_path if attachment exists
    if has_attachment and attachment_path:
        bot_message['content']['attachment']['gcs_storage_path'] = attachment_path
    
    # Only add gcs_storage_path if visualization exists
    if has_visualization and visualization_path:
        bot_message['content']['visualization']['gcs_storage_path'] = visualization_path
    
    return bot_message


def update_chat_history(chat_ref, chat_data: Dict[str, Any], new_messages: List[Dict[str, Any]]):
    """Update chat document with new messages"""
    updated_history = chat_data.get('history', [])
    updated_history.extend(new_messages)
    
    chat_ref.update({
        'history': updated_history,
        'updated_at': get_current_timestamp()
    })


# ==================== File Access Operations ====================

def get_attachment_url(db, bucket, message_id: str, user_id: str) -> Tuple[str, int]:
    """
    Generate signed URL for attachment download
    
    Args:
        db: Firestore client instance
        bucket: Cloud Storage bucket instance
        message_id: Message's unique identifier
        user_id: User's unique identifier (for ownership verification)
        
    Returns:
        Tuple of (signed_url, expires_in_seconds)
        
    Raises:
        HTTPException: 404 if attachment not found
    """
    try:
        attachment_path = find_file_path(db, user_id, message_id, 'attachment')
        
        if not attachment_path:
            raise HTTPException(status_code=404, detail="Attachment not found")
        
        signed_url = generate_signed_url(bucket, attachment_path, expiration_seconds=300)
        
        return signed_url, 300
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating attachment URL: {str(e)}"
        )


def get_visualization_url(db, bucket, message_id: str, user_id: str) -> Tuple[str, int]:
    """
    Generate signed URL for visualization (iframe)
    
    Args:
        db: Firestore client instance
        bucket: Cloud Storage bucket instance
        message_id: Message's unique identifier
        user_id: User's unique identifier (for ownership verification)
        
    Returns:
        Tuple of (signed_url, expires_in_seconds)
        
    Raises:
        HTTPException: 404 if visualization not found
    """
    try:
        visualization_path = find_file_path(db, user_id, message_id, 'visualization')
        
        if not visualization_path:
            raise HTTPException(status_code=404, detail="Visualization not found")
        
        signed_url = generate_signed_url(bucket, visualization_path, expiration_seconds=300)
        
        return signed_url, 300
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error generating visualization URL: {str(e)}"
        )


def find_file_path(db, user_id: str, message_id: str, file_type: str) -> Optional[str]:
    """
    Find file path for a message attachment or visualization
    
    Args:
        db: Firestore client instance
        user_id: User's unique identifier
        message_id: Message's unique identifier
        file_type: 'attachment' or 'visualization'
        
    Returns:
        GCS file path or None if not found
    """
    chats_ref = db.collection('chats').where('user_id', '==', user_id)
    chats = chats_ref.stream()
    
    for chat in chats:
        chat_data = chat.to_dict()
        for message in chat_data.get('history', []):
            if message['message_id'] == message_id:
                content = message.get('content', {})
                
                if file_type == 'attachment':
                    attachment = content.get('attachment', {})
                    if attachment and attachment.get('has_attachment'):
                        return attachment.get('gcs_storage_path')
                
                elif file_type == 'visualization':
                    visualization = content.get('visualization', {})
                    if visualization and visualization.get('has_visualization'):
                        return visualization.get('gcs_storage_path')
    
    return None


# def generate_signed_url(bucket, blob_path: str, expiration_seconds: int = 300) -> str:
#     """
#     Generate a signed URL for GCS object
    
#     Args:
#         bucket: Cloud Storage bucket instance
#         blob_path: Path to the blob in GCS
#         expiration_seconds: URL validity duration
        
#     Returns:
#         Signed URL string
#     """
#     if bucket is None:
#         raise HTTPException(status_code=503, detail="Storage not initialized")
    
#     blob = bucket.blob(blob_path)
#     url = blob.generate_signed_url(
#         version="v4",
#         expiration=timedelta(seconds=expiration_seconds),
#         method="GET"
#     )
#     return url
def generate_signed_url(bucket, blob_path: str, expiration_seconds: int = 300) -> str:
    """
    Generate a signed URL for GCS object
    If the file is Parquet, convert it to CSV first
    
    Args:
        bucket: Cloud Storage bucket instance
        blob_path: Path to the blob in GCS (e.g., "query_results/111/1764976697/data.parquet")
        expiration_seconds: URL validity duration
        
    Returns:
        Signed URL string for the file (CSV if converted from Parquet)
    """
    if bucket is None:
        raise HTTPException(status_code=503, detail="Storage not initialized")
    
    make_public = os.getenv('GCS_MAKE_PUBLIC', 'true').lower() == 'true'
    
    # Check if file is Parquet
    if blob_path.endswith('.parquet'):
        try:
            import pandas as pd
            import io
            
            # Download Parquet file from GCS
            parquet_blob = bucket.blob(blob_path)
            
            if not parquet_blob.exists():
                raise HTTPException(status_code=404, detail="File not found in storage")
            
            parquet_data = parquet_blob.download_as_bytes()
            
            # Convert Parquet to CSV
            df = pd.read_parquet(io.BytesIO(parquet_data))
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Upload CSV to GCS (replace .parquet with .csv)
            csv_path = blob_path.replace('.parquet', '.csv')
            csv_blob = bucket.blob(csv_path)
            csv_blob.upload_from_string(csv_data, content_type='text/csv')
            
            # Generate signed URL for CSV
            if make_public:
                return f"https://storage.googleapis.com/{bucket.name}/{csv_path}"
            else:
                url = csv_blob.generate_signed_url(
                    version="v4",
                    expiration=timedelta(seconds=expiration_seconds),
                    method="GET"
                )
                return url
                
            
        except ImportError:
            raise HTTPException(
                status_code=500, 
                detail="pandas is required for Parquet to CSV conversion. Please install it."
            )
        except Exception as e:
            raise HTTPException(
                status_code=500,
                detail=f"Error converting Parquet to CSV: {str(e)}"
            )
    
    # For non-Parquet files, generate signed URL directly
    blob = bucket.blob(blob_path)
    
    if not blob.exists():
        raise HTTPException(status_code=404, detail="File not found in storage")
    
    if make_public:
        # For uniform bucket-level access, just return the public URL
        return f"https://storage.googleapis.com/{bucket.name}/{blob_path}"
    else:
        url = blob.generate_signed_url(
            version="v4",
            expiration=timedelta(seconds=expiration_seconds),
            method="GET"
        )
        return url
