from tqdm import tqdm
from mirage.embedders import Embedder
from mirage.index.chunk_storages import ChunkStorage
from mirage.index.raw_storages import RawStorage
from .ChunkingAlgorithm import ChunkingAlgorithm
from natasha import Segmenter
from ...embedders import HuggingFaceEmbedder
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


class SemanticChunking(ChunkingAlgorithm):
    def __init__(
        self,
        raw_storage: RawStorage,
        chunk_storage: ChunkStorage,
        embedder: Embedder,
        max_chunk_size: int = 300,
        threshold: float = 0.8,
    ):
        super().__init__(raw_storage, chunk_storage)
        self.segmenter = Segmenter()
        self.max_chunk_size = max_chunk_size
        self.threshold = threshold
        self.embedder = embedder

    def chunk_a_document(self, raw_document_index, visualize=False) -> int:
        document_text: str = self.raw_storage[raw_document_index]
        if not visualize:
            sentences = [s.text for s in self.segmenter.sentenize(document_text)]
        else:
            print("Splitting text into the sentences")
            sentences = [s.text for s in tqdm(self.segmenter.sentenize(document_text))]

        if not visualize:
            embeddings = [self.embedder.embed(text=sentence) for sentence in sentences]
        else:
            print("creating embeddings of the sentences for semantic grouping")
            embeddings = [
                self.embedder.embed(text=sentence) for sentence in tqdm(sentences)
            ]

        # Вычисление попарной схожести между соседними предложениями
        similarities = []
        for i in range(1, len(embeddings)):
            similarity = cosine_similarity([embeddings[i - 1]], [embeddings[i]])[0][0]
            similarities.append(similarity)

        # Разделение на чанки с учетом семантической схожести и максимального размера
        chunks = []
        current_chunk = [sentences[0]]
        current_size = len(current_chunk[0])

        for i, similarity in enumerate(similarities):
            if (
                similarity < self.threshold
                or current_size + len(sentences[i + 1]) > self.max_chunk_size
            ):
                chunks.append(" ".join(current_chunk))
                current_chunk = [sentences[i + 1]]
                current_size = len(sentences[i + 1])
            else:
                current_chunk.append(sentences[i + 1])
                current_size += len(sentences[i + 1])

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        # Сохранение чанков в chunk_storage
        if not visualize:
            for chunk in chunks:
                self.chunk_storage.add_chunk(
                    text=chunk, raw_document_index=raw_document_index
                )
        else:
            print("Adding chunks to the storage")
            for chunk in tqdm(chunks):
                self.chunk_storage.add_chunk(
                    text=chunk, raw_document_index=raw_document_index
                )

        return 1
