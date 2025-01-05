import math
import logging

from .ChunkingAlgorithm import ChunkingAlgorithm
from ..raw_storages.RawStorage import RawStorage
from ..chunk_storages.ChunkStorage import ChunkStorage

logger = logging.getLogger(__name__)

class WordCountingChunkingAlgorithm(ChunkingAlgorithm):

    def __init__(self, raw_storage: RawStorage, chunk_storage: ChunkStorage, words_amount = 100) -> None:
        super().__init__(raw_storage, chunk_storage)
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
        logger.info(f"{len(words)} words are parsed from the document")
        for i in range(0, math.floor(len(words) / self.words_amount) - 1):

            chunk_text = ' '.join(
                words[i * 100 : min(len(words), (i + 1) * 100)]
            )
            self.chunk_storage.add_chunk(chunk_text, raw_document_index)
        return 1