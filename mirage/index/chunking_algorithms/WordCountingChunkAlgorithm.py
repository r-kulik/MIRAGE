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

    
        
    def chunk_a_document(self, raw_document_index: str) -> None:
        """
        Function that chunks a document and add it to the chunk storage

        Args:
            raw_document_index: str, link to the document in the RawStorage

        Returns:
            None, adds all chunk obtained from the document to the ChunkStorage
        """

        raw_text = self.raw_storage[raw_document_index]
        words = raw_text.split(' ')
        logger.info(f" {len(words)} words are parsed from the document")

        for i in range(0, math.floor(len(words) / self.words_amount) - 1):
            chunk_text = ' '.join(
                words[i * self.words_amount : min(len(words), (i + 1) * self.words_amount)]
            )
            self.chunk_storage.add_chunk(chunk_text, raw_document_index)
        return 1