
from mirage.index.chunk_storages import ChunkStorage
from mirage.index.chunking_algorithms import ChunkingAlgorithm
from mirage.index.raw_storages import RawStorage
from mirage.index.vector_index.VectorIndex import VectorIndex, QueryResult
from ..embedders import Embedder
from abc import abstractmethod, ABC
from typing import Any, final
from loguru import logger

class MirageIndex(ABC):

    def __init__(self, raw_storage, chunk_storage, chunking_algorithm, vector_index, visualize=False):
        super().__init__()
        self.raw_storage: RawStorage = raw_storage
        self.chunk_storage: ChunkStorage = chunk_storage
        self.chunking_algorithm: ChunkingAlgorithm = chunking_algorithm
        self.vector_index: VectorIndex = vector_index
        self.visualize = visualize