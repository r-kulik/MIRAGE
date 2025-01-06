import math
import logging

from .ChunkingAlgorithm import ChunkingAlgorithm
from ..raw_storages.RawStorage import RawStorage
from ..chunk_storages.ChunkStorage import ChunkStorage

logger = logging.getLogger(__name__)

class WordCountingChunkingAlgorithm(ChunkingAlgorithm):

    def __init__(self, raw_storage: RawStorage, chunk_storage: ChunkStorage, words_amount: int = 100) -> None:
        """
        Args:
            raw_storage: RawStorage object to pick documents from
            chunk_storage: ChunkStorage object to put in it chunks
            words_amount: number of words in each chunk
        """
        super().__init__(raw_storage, chunk_storage)
        if type(words_amount) != int or words_amount <= 0:
            raise ValueError(f"Incorrect words_amount = {words_amount} was passed in the initialization of the WorldCountingChunkingAlgorithm")
        self.words_amount = words_amount

    
        
    def chunk_a_document(self, raw_document_index: str) -> int:
        """
        Function that chunks a document and adds it to the chunk storage.

        Args:
            raw_document_index: str, link to the document in the RawStorage

        Returns:
            int: Number of chunks added to the storage
        """
        raw_text = self.raw_storage[raw_document_index]

        # Handle empty documents
        if not raw_text.strip():
            logger.warning(f"Document {raw_document_index} is empty.")
            return 0

        words = raw_text.split(' ')
        logger.info(f"{len(words)} words are parsed from the document")

        # Handle small documents
        if len(words) <= self.words_amount:
            self.chunk_storage.add_chunk(raw_text, raw_document_index)
            return 1

        for i in range(0, math.ceil(len(words) / self.words_amount)):
            chunk_text = ' '.join(
                words[i * self.words_amount : min(len(words), (i + 1) * self.words_amount)]
            )
            self.chunk_storage.add_chunk(chunk_text, raw_document_index)

        return 1