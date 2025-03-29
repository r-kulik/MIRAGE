


from abc import ABC, abstractmethod
from typing import Callable, List, Tuple
from numpy import mean, std
from pydantic import BaseModel

from mirage.embedders import Embedder
from mirage.index import QueryResult
from mirage.index.chunk_storages import ChunkStorage

class Reranker(ABC):
    """Performs a Rerunking functionality
    """

    def __init__(self):
        super().__init__()

    @staticmethod
    def minmax_score_normalization(search_results: List[QueryResult]) -> List[QueryResult]:
        if not search_results: return []
        minmax_norm: Callable[[float, float, float], float] = lambda x, minx, maxx: (x - minx) / (maxx - minx)
        min_score = min(result.score for result in search_results)
        max_score = max(result.score for result in search_results)
        return [
            QueryResult(
                score=0 if min_score == max_score else minmax_norm(result.score, min_score, max_score),
                chunk_storage_key=result.chunk_storage_key,
                vector=result.vector
            )
            for result in search_results
        ]
    
    @staticmethod
    def score_standartization(search_results: List[QueryResult]) -> List[QueryResult]:
        if not search_results: return []
        search_scores = [query_result.score for query_result in search_results]
        mean_score = mean(search_scores)
        std_score = std(search_scores)
        vector_search_results_normalized = [
            QueryResult(
                score=((query_result.score - mean_score) / std_score) if std_score != 0 else 0,
                chunk_storage_key=query_result.chunk_storage_key,
                vector=query_result.vector
            )
            for query_result in search_results
        ]

    @abstractmethod
    def __call__(
        self, fulltext_search_results: List[QueryResult], 
        vector_search_results: List[QueryResult]
        ) -> List[QueryResult]:
        """Execution of the Rank Fusion algorithm

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