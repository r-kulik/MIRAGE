from typing import Optional, Self

from mirage.index import QueryResult
from .Reranker import Reranker


class LinearCombinationReranker(Reranker):
    fulltext_score_weight: float
    vector_score_weight: float

    def __init__(self, fulltext_score_weight: float, vector_score_weight: float):
        super().__init__()
        self.fulltext_score_weight = fulltext_score_weight
        self.vector_score_weight = vector_score_weight

    def __call__(self, fulltext_search_results, vector_search_results):
        results_dict = {}
        for query_result in vector_search_results:
            if query_result.chunk_storage_key in results_dict:
                raise ValueError('You are trying to add a duplicate chunk thorugh a vector search')
            results_dict[query_result.chunk_storage_key] = query_result.score * self.vector_score_weight
        for query_result in fulltext_search_results:
            results_dict[query_result.chunk_storage_key] = results_dict.get(
                query_result.chunk_storage_key,
                0
            ) + query_result.score * self.fulltext_score_weight
        return [
            QueryResult(
                score=combined_score,
                chunk_storage_key=chunk_storage_key
            )
            for chunk_storage_key, combined_score in sorted(
                results_dict.items(),
                key=lambda x: x[1],
                reverse=True
            )
        ]    