from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable, final
import numpy as np
from .TextNormalizer import TextNormalizer
from ..index.chunk_storages import ChunkStorage
from ..index.vector_index.VectorIndex import VectorIndex
import logging
import tqdm


class EmbedderIsNotTrainedException(Exception):
    """This exception must be raised when embed(), process_chunks() or convert_chunks_to_vector_index() is called without training on corpora
    on all vectorizers that need to be trained
    """
    def __init__(self, additional_message = None):
        super().__init__(f"""
You are trying to infere an embedder that has not been trained on a corpora. Try fit(chunk: ChunkSotrage) method to solve this problem 
{'Additional Info: ' if additional_message is not None else ''} {additional_message}
""")


class Embedder(ABC):
    def __init__(self, normalizer: Optional[TextNormalizer] | Callable[[str], str] | bool = None):
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
            normalizer = TextNormalizer(stop_word_remove=True, word_generalization="stem")
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
        """Train a embedder to operate on your corpora. For td-idf and BoW embedders is necessary to start operating
        
        Parameters
        ----------
        chunks : ChunkStorage
            Object of Chunk storage to take the chunks for training from
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
        if self.normalizer is None or self.normalizer == False:
            return text
        if type(self.normalizer) == callable:
            return self.normalizer(text)
        elif type(self.normalizer) == TextNormalizer:
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
    

    @final
    def process_chunks(self, chunks: ChunkStorage) -> Dict[str, np.ndarray]:
        """
        Обработка чанков и преобразование их в словарь векторов.
        Если модель не обучена, вызывается метод fit для обучения на переданных чанках.

        Params
        ------------
            chunks: ChunkStorage 
                Хранилище чанков.

        Returns
        ------------
            vectors: Dict[int, np.ndarray]
                Словарь, где ключ — идентификатор чанка, а значение — векторное представление чанка.

        Raises
        -----------
            EmbedderIsNotTrainedException
                an exception you see if the fit() method on a vectorizer that needs fitting was never called
        """
        if not self.is_fitted:
            raise EmbedderIsNotTrainedException(additional_message="while process_chunks() method was called")
        vectors = {}
        for chunk_key, chunk in chunks:  # Предполагаем, что ChunkStorage поддерживает метод __items__()
            vector = self.embed(chunk)  # Преобразуем каждый чанк в вектор
            vectors[chunk_key] = vector
        return vectors
    
    @final
    def convert_chunks_to_vector_index(
            self,
            chunk_storage: ChunkStorage,
            vector_index: VectorIndex,
            visualize = False
    ) -> None:
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

        if not self.is_fitted:
            raise EmbedderIsNotTrainedException

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
        
        if not vector_index.is_trained:
            vector_index.train()