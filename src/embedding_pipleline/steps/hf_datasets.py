from datasets import (Dataset, load_dataset,  # type: ignore
                      load_dataset_builder)
from zenml import step
from zenml.logger import get_logger

from settings import settings

logger = get_logger(__name__)

@step 
def load_hf_dataset(dataset_name: str = "lansinuote/simple_facenet") -> Dataset:
    """Loads and returns the HuggingFace dataset for """