# ChromaDB configuration and embedding logic
# Manages persistent vector database storage
import chromadb
import os
from .logger import setup_logger

# Initialize logger for this module
logger = setup_logger(__name__)


def get_chromadb_client() -> chromadb.Client:
    """Initialize persistent ChromaDB client"""
    # get path from .env 
    persist_dir = os.getenv("CHROMA_PERSIST_DIRECTORY", "./vector_db")
    
    logger.info(f"Initializing ChromaDB client with persist directory: {persist_dir}")
    
    try:
        # intitialize chromadb persistent client
        client = chromadb.PersistentClient(path=persist_dir)
        logger.info("ChromaDB client initialized successfully")
        return client
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
        raise

def get_or_create_collection(collection_name: str = None) -> chromadb.Collection:
    """Get existing collection or create if doesn't exist"""
    if not collection_name:
        collection_name = os.getenv("COLLECTION_NAME", "default_collection")

    logger.info(f"Accessing collection: {collection_name}")
    
    try:
        client = get_chromadb_client()
        # Use get_or_create_collection method instead of try/except
        collection = client.get_or_create_collection(name=collection_name)
        logger.info(f"Retrieved or created collection: {collection_name}")
        
        return collection
    except Exception as e:
        logger.error(f"Failed to get or create collection '{collection_name}': {e}")
        raise

