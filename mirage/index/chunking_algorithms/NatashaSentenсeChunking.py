from mirage.index.chunk_storages import ChunkStorage
from mirage.index.raw_storages import RawStorage
from .ChunkingAlgorithm import ChunkingAlgorithm
from natasha import Segmenter


class NatashaSentenceChunking(ChunkingAlgorithm):
    def __init__(self, raw_storage: RawStorage, chunk_storage: ChunkStorage, sentences_in_chunk: int = 1):
        super().__init__(raw_storage, chunk_storage)
        self.segmenter = Segmenter()
        self.sentences_in_chunk = sentences_in_chunk
    
    def chunk_a_document(self, raw_document_index) -> int:
        document_text: str = self.raw_storage[raw_document_index]
        sentences = list(self.segmenter.sentenize(document_text))
        for i in range(len(sentences) // self.sentences_in_chunk):
            self.chunk_storage.add_chunk(
                text=' '.join(
                    map(
                        lambda x: x.text,
                        sentences[i * self.sentences_in_chunk: (i + 1) * self.sentences_in_chunk]
                    )
                ),
                raw_document_index=raw_document_index
            )
        return 1
        