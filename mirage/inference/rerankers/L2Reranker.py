from math import sqrt
from typing import Callable, Dict, List

from mirage.index import QueryResult
from .Reranker import Reranker


class L2Reranker(Reranker):
    """This Reranker uses a Euclidian distance to point ||(Bm25(d, q), sim(d, q))||_2 as a score of relevance
    
    Values of original scores can be or can not be min-max normalized
    """
    def __init__(self, minmax_normalize: bool = False):
        """Initialization of L2Reranker

        Parameters
        ----------
        min_max_normalization : bool, optional
            Whether or not scores of the original ranking will be minmax normalized, by default False
        """
        super().__init__()
        self.minmax_normalize = minmax_normalize

    def __call__(self, fulltext_search_results: List[QueryResult], vector_search_results: List[QueryResult]) -> List[QueryResult]:
        # -------------------------------------------
        # normalization of scores if needed
        if self.minmax_normalize:
           fulltext_search_results = Reranker.minmax_score_normalization(fulltext_search_results)
           vector_search_results = Reranker.minmax_score_normalization(vector_search_results)
        # -------------------------------------------
        # results_dict contains pairs of {chunk_storage_key: relevance_score}
        results_dict: Dict[str, float] = {}
        # -------------------------------------------
        # adding all vector search results to the dictionary
        for query_result in vector_search_results:
            if query_result.chunk_storage_key in results_dict:
                raise ValueError('You are trying to add a duplicate chunk thorugh a vector search')
            results_dict[query_result.chunk_storage_key] = query_result.score
        # -------------------------------------------
        # iterating thorugh results from the fulltext search
        # creating new entries if necessary, or adding fulltext relevance
        # to the existing entires in the results_dict
        for query_result in fulltext_search_results:
            results_dict[query_result.chunk_storage_key] = sqrt(
                results_dict.get(
                    query_result.chunk_storage_key,
                    0
                ) ** 2 + query_result.score ** 2
            )
        # -------------------------------------------
        # sorting items in the dict to present them in the descending order of relevance
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