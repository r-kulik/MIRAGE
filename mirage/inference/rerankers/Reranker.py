


from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

from pydantic import BaseModel

from mirage.embedders import Embedder
from mirage.index import QueryResult
from mirage.index.chunk_storages import ChunkStorage

class Reranker(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(
        self, fulltext_search_results: List[QueryResult], 
        vector_search_results: List[QueryResult]
        ) -> List[QueryResult]:
        pass