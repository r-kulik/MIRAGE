from typing import Dict, Optional, Self

from mirage.index import QueryResult
from .Reranker import Reranker


class LinearCombinationReranker(Reranker):
    """Reranker based on linear combination of two scores
    score(d) = K1 * BM25(d, q) + K2 * sim(d, q)\n\n
    where:
        * d - document
        * q - user query
        * K1 - settable hyperparameter (fulltext_score_weight)
        * K2 - settable hyperparameter (vector_score_weight)
        * BM25(d, q) - score of document after fulltext_search of query
        * sim(d, q) - cosine similarity between document vector and query vector
    """
    fulltext_score_weight: float
    vector_score_weight: float

    def __init__(self, fulltext_score_weight: float, vector_score_weight: float,):
        """Creation of linear combination reranker

        Parameters
        ----------
        fulltext_score_weight : float
            Multiplicator of the score obtained from BM-25 algorithm
        vector_score_weight : float
            Multiplicator of the cosine similarity between query and document vectors
        """
        super().__init__()
        self.fulltext_score_weight = fulltext_score_weight
        self.vector_score_weight = vector_score_weight

    def __call__(self, fulltext_search_results, vector_search_results):
        # -------------------------------------------
        # results_dict contains pairs of {chunk_storage_key: relevance_score}
        results_dict: Dict[str, float] = {}
        # -------------------------------------------
        # adding all vector search results to the dictionary
        for query_result in vector_search_results:
            if query_result.chunk_storage_key in results_dict:
                raise ValueError('You are trying to add a duplicate chunk thorugh a vector search')
            results_dict[query_result.chunk_storage_key] = query_result.score * self.vector_score_weight
        # -------------------------------------------
        # iterating thorugh results from the fulltext search
        # creating new entries if necessary, or adding fulltext relevance
        # to the existing entires in the results_dict
        for query_result in fulltext_search_results:
            
            results_dict[query_result.chunk_storage_key] = results_dict.get(
                query_result.chunk_storage_key,
                0
            ) + query_result.score * self.fulltext_score_weight
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