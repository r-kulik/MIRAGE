from typing import Self
from mirage.embedders.Embedder import Embedder
from mirage.inference.MirageInfefrence import MirageInference
from .index.MirageIndex import MirageIndex


class MIRAGE:
    def __init__(
        self, index: MirageIndex, inference=MirageInference, embedder=Embedder
    ) -> None:
        self.index: MirageIndex = index
        self.inference: MirageInference = inference
        self.embedder: Embedder = embedder

    def create_index(self):
        self.index.chunking_algorithm.execute()
        self.embedder.convert_chunks_to_vector_index(
            chunk_storage=self.index.chunk_storage, vector_index=self.index.vector_index
        )

    def save(self, filepath: str): ...

    def load(self, filepath: str): ...

    def query(self, query_text):
        return self.inference.llm_adapter.do_request(
            query_text,
            chunk_storage=self.index.chunk_storage,
            indexes=[
                result.chunk_storage_key
                for result in self.inference.reranker(
                    fulltext_search_results=(
                        self.index.chunk_storage.query(query=query_text)
                        if self.inference.quorum is None
                        else self.inference.quorum.query(text=query_text)
                    ),
                    vector_search_results=self.index.vector_index.query(
                        query_vector=self.embedder.embed(text=query_text)
                    ),
                )
            ],
        )

    def init_from_experimental_json(json_text: str) -> Self: ...
