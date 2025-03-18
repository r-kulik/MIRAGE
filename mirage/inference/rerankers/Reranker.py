


from abc import ABC, abstractmethod
from typing import Callable, List, Tuple

from pydantic import BaseModel

from mirage.embedders import Embedder
from mirage.index.chunk_storages import ChunkStorage


class RankingDTO(BaseModel):
    score: float
    chunk: ChunkStorage.ChunkNote

class Reranker(ABC):
    fulltext_scoring_function: Callable[[str, str], float]
    vector_scoring_function: Callable[[str, str], float]
    embedder: Embedder

    def __init__(
            self, fulltext_scoring_function: Callable,
            vector_scoring_function: Callable[[str, str], float],
            embedder: Embedder):
        super().__init__()
        self.fulltext_scoring_function = fulltext_scoring_function
        self.vector_scoring_function = vector_scoring_function
        self.embedder = embedder

    @abstractmethod
    def rerank(
        self, query: str, chunk_notes: List[ChunkStorage.ChunkNote]
    ) -> List[RankingDTO]:
        """This function performs reranking of the ChunkNote object related to the
        query, asked by user

        Parameters
        ----------
        query : str
            User query, chunks are related to
        chunk_notes : List[ChunkStorage.ChunkNote]
            List of ChunkNotes obtained from the fulltext and semantic search

        Returns
        -------
        List[RankingDTO]
            List of chunk notes (rankingDTO.chunk) with weights (rankingDTO.weight)
        """
        pass



