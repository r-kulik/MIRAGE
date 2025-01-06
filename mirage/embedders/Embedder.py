from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable
import numpy as np
from .TextNormallizer import TextNormallizer
from ..index.chunk_storages import ChunkStorage
from ..index.vector_index.VectorIndex import VectorIndex
import logging
import tqdm

class Embedder(ABC):
    def __init__(self, normalizer: Optional[TextNormallizer] | Callable[[str], str] | bool = None):
        """
        Инициализация Embedder.

        Args:
            normalizer (TextNormallizer | bool | Callable | None) Text normalizer. 
                If None or False, no normalization for text is needed
                If True standard mirage.embedders.TextNormalizer is applied
                Any TextNormalizer inherited object is allowed
                Any function: str -> str that normalize text is allowed
        """
        if type(normalizer) == bool and normalizer:
            normalizer = TextNormallizer(stop_word_remove=True, word_generalization="stem")
        self.normalizer = normalizer
        self._dim: int = -1   # This field MUST be ovverided by all realizations of Embedder
        self.is_fitted = True # False for all realizations that must be retrained

    def get_dimensionality(self) -> int:
        """
        Returns the dimensionality of the vectors that will be obtained by the embedder
        """
        return self._dim
    
    @abstractmethod
    def fit(chunks: ChunkStorage) -> None:
        """
        This function must be overriden for all embedding algorithms that need additional training before embedding
        """
        raise NotImplementedError

    def _normalize(self, text: str) -> str:
        """
        Нормализация текста с использованием нормализатора, если он задан.

        Args:
            text (str): Входной текст.

        Returns:
            str: Нормализованный текст.
        """
        if self.normalizer is None:
            return text
        if type(self.normalizer) == callable:
            return self.normalizer(text)
        elif type(self.normalizer) == TextNormallizer:
            return self.normalizer.normalize(text)

    

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """
        Абстрактный метод для получения векторного представления текста.

        Args:
            text (str): Входной текст.

        Returns:
            np.ndarray: Векторное представление текста в виде массива numpy.
        """
        raise NotImplementedError
    
    def process_chunks(self, chunks: ChunkStorage) -> Dict[int, np.ndarray]:
        """
        Обработка чанков и преобразование их в словарь векторов.
        Если модель не обучена, вызывается метод fit для обучения на переданных чанках.

        Args:
            chunks (ChunkStorage): Хранилище чанков.

        Returns:
            Dict[int, np.ndarray]: Словарь, где ключ — идентификатор чанка, а значение — векторное представление чанка.
        """
        # print("PROCESS CHUNKS FUNCTION HAS BEEN REACHED")
        vectors = {}
        for chunk_key, chunk in chunks:  # Предполагаем, что ChunkStorage поддерживает метод __items__()
            # print(chunk_key)
            vector = self.embed(chunk)  # Преобразуем каждый чанк в вектор
            vectors[chunk_key] = vector
        return vectors
    

    def convert_chunks_to_vector_index(
            self,
            chunk_storage: ChunkStorage,
            vector_index: VectorIndex,
            visualize = False
    ) -> VectorIndex:
        '''
        This function automatically "populate" a VectorIndex object with the vectors obtained from the chunks from ChunkStorage object
        Args:
            chunk_storage: ChunkStorage object that is considered as a source of text
            vector_index: VectorIndex object that will be filled with vectors obtained from the chunks using the Embedder (self)

        ```py
        >>> chunks = ChunkStorage()
        >>> vidx = VectorIndex()
        >>> emb = Embedder()
        >>> emb.convert_chunks_to_vector_index(chunks, vidx)
        ```        
        '''

        if visualize:
            print("Converting ChunkStorage to VectorIndex")
            progress_bar = tqdm.tqdm(total=len(chunk_storage.get_indexes()))
        
        for chunk_storage_key, chunk_text in chunk_storage:
            vector_of_chunk = self.embed(chunk_text)
            vector_index.add(
                vector=vector_of_chunk,
                chunk_storage_key=chunk_storage_key
            )
            if visualize: progress_bar.update(1)