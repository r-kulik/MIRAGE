import math
from loguru import logger

from .ChunkingAlgorithm import ChunkingAlgorithm
from ..raw_storages.RawStorage import RawStorage
from ..chunk_storages.ChunkStorage import ChunkStorage


class WordCountingChunkingAlgorithm(ChunkingAlgorithm):

    def __init__(self, raw_storage: RawStorage, chunk_storage: ChunkStorage, words_amount: int = 100, overlap: float = 0.1) -> None:
        """
        Args:
            raw_storage: RawStorage object to pick documents from
            chunk_storage: ChunkStorage object to put chunks in it
            words_amount: number of words in each chunk
            overlap: fraction of overlap between chunks (e.g., 0.1 for 10% overlap)
        """
        super().__init__(raw_storage, chunk_storage)
        if type(words_amount) != int or words_amount <= 0:
            raise ValueError(f"Incorrect words_amount = {words_amount} was passed in the initialization of the WordCountingChunkingAlgorithm")
        if not (0 <= overlap <= 1):
            raise ValueError(f"Incorrect overlap = {overlap} was passed in the initialization of the WordCountingChunkingAlgorithm. It should be between 0 and 1.")
        self.words_amount = words_amount
        self.overlap = overlap

    def chunk_a_document(self, raw_document_index: str) -> int:
        logger.info(f'Reading a document... {raw_document_index}')
        assert type(raw_document_index) == str

        raw_text = self.raw_storage[raw_document_index]
        # Handle empty documents
        if not raw_text.strip():
            logger.warning(f"Document {raw_document_index} is empty.")
            return 0

        words = raw_text.split(' ')
        # logger.info(f"{len(words)} words are parsed from the document")

        # Handle small documents
        if len(words) <= self.words_amount:
            self.chunk_storage.add_chunk(raw_text, raw_document_index)
            return 1

        # Calculate the number of words in the overlap
        overlap_size = math.floor(self.words_amount * self.overlap)
        # logger.info(f"Using an overlap size of {overlap_size} words")

        # Chunking the document with overlap
        chunks_count = 0
        for i in range(0, len(words), self.words_amount - overlap_size):
            chunk_text = ' '.join(
                words[i : min(i + self.words_amount, len(words))]
            )
            self.chunk_storage.add_chunk(chunk_text, raw_document_index)
            chunks_count += 1

        return chunks_count