# ChromaDB client and collection management.
import chromadb
from . import config
from .logger import setup_logger

logger = setup_logger(__name__)


def get_chromadb_client() -> chromadb.ClientAPI:
    """Return a persistent ChromaDB client."""
    try:
        return chromadb.PersistentClient(path=config.CHROMA_PERSIST_DIRECTORY)
    except Exception as e:
        logger.error(f"Failed to initialize ChromaDB client: {e}")
        raise


def get_or_create_collection(collection_name: str = None) -> chromadb.Collection:
    """Get an existing collection, creating it if necessary."""
    name = collection_name or config.COLLECTION_NAME
    try:
        client = get_chromadb_client()
        return client.get_or_create_collection(name=name)
    except Exception as e:
        logger.error(f"Failed to get or create collection '{name}': {e}")
        raise
