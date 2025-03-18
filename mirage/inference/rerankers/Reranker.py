


from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

from pydantic import BaseModel

from mirage.embedders import Embedder
from mirage.index import QueryResult
from mirage.index.chunk_storages import ChunkStorage

class Reranker(ABC):
    """Performs a Rerunking functionality
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def __call__(
        self, fulltext_search_results: List[QueryResult], 
        vector_search_results: List[QueryResult]
        ) -> List[QueryResult]:
        """Executopn of the Rank Fusion algorithm

        Parameters
        ----------
        fulltext_search_results : List[QueryResult]
            Results obtained from the fulltext search
        vector_search_results : List[QueryResult]
            Results obtained from the vector search

        Returns
        -------
        List[QueryResult]
            Results ordered in the descending order according to the relevance score
        """
        pass