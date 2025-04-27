from math import ceil, floor
from mirage.index.chunk_storages import ChunkStorage
from mirage.index.raw_storages import RawStorage
from .ChunkingAlgorithm import ChunkingAlgorithm
from natasha import Segmenter


class NatashaSentenceChunking(ChunkingAlgorithm):
    def __init__(self, raw_storage: RawStorage, chunk_storage: ChunkStorage, sentences_in_chunk: int = 1, overlap: float = 0):
        super().__init__(raw_storage, chunk_storage)
        self.segmenter = Segmenter()
        self.sentences_in_chunk = sentences_in_chunk
        self.overlap = overlap
        self.overlap_size_in_sentences = floor(self.sentences_in_chunk * self.overlap)
    
    def chunk_a_document(self, raw_document_index) -> int:
        document_text: str = self.raw_storage[raw_document_index]
        sentences = list(self.segmenter.sentenize(document_text))
        for i in range(0, len(sentences),  self.sentences_in_chunk - self.overlap_size_in_sentences):
            self.chunk_storage.add_chunk(
                text=' '.join(
                    map(
                        lambda x: x.text,
                        sentences[i : i + self.sentences_in_chunk]
                    )
                ),
                raw_document_index=raw_document_index
            )
        return 1
        