from mirage.index import RawStorage, ChunkStorage, ChunkingAlgorithm, VectorIndex, QueryResult
from mirage.embedders import Embedder
from abc import abstractmethod, ABC
from typing import final


class MirageIndex(ABC):

    def __init__(self, raw_storage, chunk_storage, chunking_algorithm, embedder, vector_index, visualize=False):
        super().__init__()
        self.raw_storage: RawStorage = raw_storage
        self.chunk_storage: ChunkStorage = chunk_storage
        self.chunking_algorithm: ChunkingAlgorithm = chunking_algorithm
        self.embedder: Embedder = embedder
        self.vector_index: VectorIndex = vector_index
        self.visualize = visualize

    @final
    def create_index(self, *args, **kwargs):
        if self.visualize: print('Performing chunking algorithm')
        doduments_processed = self.chunking_algorithm.execute(visualize=self.visualize)
        if self.visualize: print(f"Processed {doduments_processed} documents")
        if not self.embedder.is_fitted:
            if self.visualize: print("Training an embedder...")
            self.embedder.fit(self.chunk_storage)                               # Training the embeedder if it is not trained
            self.vector_index.dim = self.embedder.get_dimensionality()          # Providing dimensionality of the embedder to the vector index
            if self.visualize: print("Embedder trained")
        if self.visualize: print("Converting chunks to vectors")
        self.embedder.convert_chunks_to_vector_index(self.chunk_storage, self.vector_index, visualize=self.visualize)
        if self.visualize: print("Creation of index has been done")          
    
    @final
    def query(self, query: str, top_k: int) -> list[str]:
        embedded_query = self.embedder.embed(query)
        return self.vector_index.query(embedded_query, top_k=top_k)