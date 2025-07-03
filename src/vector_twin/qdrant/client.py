from functools import lru_cache
from qdrant_client import QdrantClient 
from vector_twin.settings import settings 

@lru_cache(maxsize=1)
def get_qdrant_client(use_qdrant_cloud: bool = True) -> QdrantClient: 
    """Gets a configured qdrant client instance. 
    
    Args: 
    use_qdrant_cloud(bool, optional): Whether to connect Qdrant Cloud or local instance 
    defaults to true 

    Returns: 
    QdrantClient: Configured Qdranr client connected to either cloud of local instance. 
    
    """

    if use_qdrant_cloud: 
        return QdrantClient(
            url=settings.QDRANT_CLOUD_URL,
            api_key=settings.QDRANT_CLOUD_API_KEY,
            port=settings.QDRANT_PORT,
        )
    else: 
        return QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )