from typing import Optional, Self
from .Reranker import Reranker


class LinearCombinationReranker(Reranker):
    fulltext_coeffitient: float
    vector_coeffitient: float

    def __init__(
            self, fulltext_scoring_function, 
            vector_scoring_function, embedder,
            fulltext_coeffitient: float = 0.5, 
            vector_coeffitient: float = 0.5
            ) -> Self:
        super().__init__(fulltext_scoring_function, 
                         vector_scoring_function, 
                         embedder)
        self.fulltext_coeffitient = fulltext_coeffitient
        self.vector_coeffitient = vector_coeffitient
    
