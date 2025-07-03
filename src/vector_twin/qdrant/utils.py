import logging 
from qdrant_client import QdrantClient 
import qdrant_client
from qdrant_client.http.models import PointStruct 
from qdrant_client.http.models import Distance, VectorParams  

from vector_twin.settings import settings 
logger = logging.getLogger(__name__)

def create_qdrant_collection(client: QdrantClient, 
                             collection_name: str = settings.QDRANT_COLLECTION_NAME, 
                             vector_dimensions: int = settings.QDRANT_VECTOR_DIMENSIONS): 
    """Creates a new Qdrant collection with the given name and vector dimensions. 
    
    Args: 
    client(QdrantClient): Qdrant client instance 
    collection_name(str, optional): Name of the collection to create, defaults to settings.QDRANT_COLLECTION_NAME 
    vector_dimensions(int, optional): Number of dimensions for the vector space, defaults to settings.QDRANT_VECTOR_DIMENSIONS 
    
    """
    if not qdrant_client.collection_exists(collection_name): 
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_dimensions, distance=Distance.COSINE),
        )

def insert_image_embedding(
        qdrant_client: QdrantClient,
        img_embedding: list[float],
        img_id: str,
        img_label: str,
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
): 
    """Inserts or updates an image embedding in the specified Qdrant collection.

    Args:
        qdrant_client (QdrantClient): The Qdrant client instance to use.
        img_embedding (list[float], optional): Vector embedding of the image.
            Defaults to None.
        img_id (str, optional): Unique identifier for the image.
            Defaults to None.
        img_label (str, optional): Label/metadata for the image.
            Defaults to None.
        collection_name (str, optional): Name of collection to insert into.
            Defaults to settings.QDRANT_COLLECTION_NAME.
    """
    try: 
        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=img_id,
                    vector=img_embedding,
                    payload={
                        "label": img_label,
                    }
                )
            ]
        )
    except Exception as e: 
        logger.error(f"Error inserting image embedding: {e}") 

def get_top_k_similar_images(
        qdrant_client: QdrantClient,
        query_embedding: list[float],
        collection_name: str = settings.QDRANT_COLLECTION_NAME,
        k: int = 5,
): 
    """Retreives k most similar images to the query embedding. 
    
    Args: 
    qdrant_client(QdrantClient): Qdrant client instance 
    query_embedding(list[float]): Vector embedding of the query image 
    collection_name(str, optional): Name of the collection to search in, defaults to settings.QDRANT_COLLECTION_NAME 
    k(int, optional): Number of similar images to retrieve, defaults to 5 
    
    Returns: 
    list[dict]: List of dictionaries containing image metadata and similarity score 
    """
    return qdrant_client.search(
        collection_name=collection_name,
        query_vector=query_embedding,
        limit=k
    )
