from typing import List

from mirage.index import QueryResult
from .Reranker import Reranker


class BordaScoreReranker(Reranker):
    """
    Ranks two arrays in order of decreasing the BordaScore
    BordaScore is defined as a sum of indexes of element counting from the end of the each list
    """

    def __init__(self):
        super().__init__()

    def __call__(
        self,
        fulltext_search_results: List[QueryResult],
        vector_search_results: List[QueryResult],
    ) -> List[QueryResult]:
        combined = {
            result.chunk_storage_key: len(fulltext_search_results) - 1 - i
            for i, result in enumerate(fulltext_search_results)
        }
        for i, result in enumerate(vector_search_results):
            combined[result.chunk_storage_key] = (
                combined.get(result.chunk_storage_key, 0)
                + len(vector_search_results)
                - 1
                - i
            )
        return list(
            sorted(
                (
                    QueryResult(
                        score=combined[chunk_storage_key],
                        chunk_storage_key=chunk_storage_key,
                        vector=None,
                    )
                    for chunk_storage_key in combined
                ),
                key=lambda x: x.score,
                reverse=True,
            )
        )
