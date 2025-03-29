from collections import namedtuple
from typing import Dict, List
from mirage.index import QueryResult
from .Reranker import Reranker


class WeightedProductReranker(Reranker):
    """Reranking based on Vector Product

    score(d, q) = BM25(d, q) ^ a * sim(d, q) * b
    """

    def __init__(self, fulltext_power: float = 1, vector_power: float = 1, minmax_normalize: bool = False):
        super().__init__()
        self.fulltext_power = fulltext_power
        self.vector_power = vector_power
        self.minmax_normalize = minmax_normalize

    def __call__(self, fulltext_search_results: List[QueryResult], vector_search_results: List[QueryResult]) -> List[QueryResult]:
        # -----------------------------------------------------
        # normzalization of scores if needed
        if self.minmax_normalize:
            fulltext_search_results = Reranker.minmax_score_normalization(fulltext_search_results)
            vector_search_results = Reranker.minmax_score_normalization(vector_search_results)
        # -----------------------------------------------------
        ScorePair = namedtuple('ScorePair', ['fulltext_score', 'vector_score'])
        combined_results: Dict[str, ScorePair] = {}
        for result in vector_search_results:
            if result.chunk_storage_key in combined_results:
                raise ValueError('There is an attempt to add a single chunk twice in rank fusion')
            combined_results[result.chunk_storage_key] = ScorePair(fulltext_score=0, vector_score=result.score)
        for result in fulltext_search_results:
            vector_score = 0
            if result.chunk_storage_key in combined_results:
                vector_score = combined_results[result.chunk_storage_key].vector_score
            combined_results[result.chunk_storage_key] = ScorePair(fulltext_score=result.score, vector_score=vector_score)
        return list(
            sorted(
                [
                    QueryResult(
                        score = (score_pair.fulltext_score ** self.fulltext_power * score_pair.vector_score ** self.vector_power),
                        chunk_storage_key=chunk_storage_key,
                        vector=None
                    )
                    for chunk_storage_key, score_pair in combined_results.items()
                ],
                key=lambda x: x.score,
                reverse=True
            )
        )