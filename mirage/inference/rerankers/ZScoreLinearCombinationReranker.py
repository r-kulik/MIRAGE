from typing import List
from mirage.index import QueryResult
from .Reranker import Reranker
from .LinearCombinationReranker import LinearCombinationReranker
from numpy import mean, std


class ZScoreLinearCombinationReranker(LinearCombinationReranker):
    """Normalized Linear Combination Reranker using the Z-score

    score(d, q) = K1 * n ( BM25 (d, q) ) + K2 * n ( sim (d, q))

    where:
        * d - document
        * q - user query
        * K1 - settable hyperparameter (fulltext_score_weight)
        * K2 - settable hyperparameter (vector_score_weight)
        * BM25(d, q) - score of document after fulltext_search of query
        * sim(d, q) - cosine similarity between document vector and query vector
        * n - normalization function such that n(x) = (x - mean(x)) / (std (x))
    """

    def __init__(self, fulltext_score_weight, vector_score_weight):
        super().__init__(
            fulltext_score_weight=fulltext_score_weight,
            vector_score_weight=vector_score_weight,
        )

    def __call__(
        self,
        fulltext_search_results: List[QueryResult],
        vector_search_results: List[QueryResult],
    ) -> List[QueryResult]:
        fulltext_search_results_normalized = Reranker.score_standartization(
            search_results=fulltext_search_results
        )
        vector_search_results_normalized = Reranker.score_standartization(
            search_results=vector_search_results
        )
        return super().__call__(
            fulltext_search_results=fulltext_search_results_normalized,
            vector_search_results=vector_search_results_normalized,
        )
