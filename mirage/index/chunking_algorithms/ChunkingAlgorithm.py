
from ..raw_storages.RawStorage import RawStorage
from ..chunk_storages.ChunkStorage import ChunkStorage
from abc import abstractmethod

import tqdm
import logging
logger = logging.getLogger(__name__)



class ChunkingAlgorithm:
    """
    A chunking algorithm is a procedure that creates a storage of chunks from the text documents in the RawStorage
    """

    def __init__(self, raw_storage: RawStorage, chunk_storage: ChunkStorage) -> None:
        """
        Chunking algorithms realizations must take raw_storage, chunk_storage only. Other parameters must be optional with the provided default value
        """
        self.raw_storage = raw_storage
        self.chunk_storage = chunk_storage

    @abstractmethod
    def chunk_a_document(raw_docuemnt_index) -> int:
        raise NotImplementedError


    def execute(self, visualize=False) -> None:
        self.chunk_storage.clear()
        logger.info("Chunking of documents")
        raw_document_indexes = self.raw_storage.get_indexes()
        parsed_indexes = set(
            self.chunk_storage.get_raw_index_of_document(index) for index in  self.chunk_storage.get_indexes()
        )
        return sum([
            self.chunk_a_document(raw_document_index)
            for raw_document_index in 
            raw_document_indexes if not raw_document_index in parsed_indexes
        ])
    
