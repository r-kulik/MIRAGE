from collections import namedtuple
from typing import Dict, List
from mirage.index import QueryResult
from .Reranker import Reranker


class ReciprocalRankFusionReranker(Reranker):
    """Reciprocal Rank Fusion combines two ranked lists considering the ordering of the list

    RRF(d) = (r_v(d) + c) ** - 1 + (r_f(d) + c) ** -1

    where:
        * c is a hyperparameter (equal to 60)
        * r_v(d) index of document in a vector search ranked list
        * r_f(d) index of document in a fulltext search ranked list
    """

    def __init__(self, c: int = 60):
        super().__init__()
        self.c = c

    def __call__(self, fulltext_search_results: List[QueryResult], vector_search_results: List[QueryResult]) -> List[QueryResult]:
        RankPair = namedtuple('RankPair', ['fulltext_rank', 'vector_rank'])
        results: Dict[str, RankPair] = {}
        # -----------------------------------------------
        # adding in the end of the list elements which are not presented in both lists
        for i, result in enumerate(vector_search_results):
            try:
                j = fulltext_search_results.index(result)
            except ValueError:
                fulltext_search_results.append(result)
                j = len(fulltext_search_results) - 1
            results[result.chunk_storage_key] = RankPair(fulltext_rank=j, vector_rank=i)
        for j, result in enumerate(fulltext_search_results):
            if result.chunk_storage_key not in results:
                vector_search_results.append(result)
                results[result.chunk_storage_key] = RankPair(fulltext_rank=j, vector_rank=len(vector_search_results) - 1)
        return list(
            sorted(
                (QueryResult(
                    chunk_storage_key=chunk_storage_key, vector=None,
                    score=(1 / (rank_pair.vector_rank + self.c)) + (1 / (rank_pair.fulltext_rank + self.c))
                )
                for chunk_storage_key, rank_pair in results.items()),
                key=lambda x: x.score,
                reverse=True
            )
        )
        