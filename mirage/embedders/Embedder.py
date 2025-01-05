from abc import ABC, abstractmethod
from typing import Optional, Dict, Callable
import numpy as np
from .TextNormallizer import TextNormallizer
from ..index.chunk_storages import ChunkStorage

class Embedder(ABC):
    def __init__(self, normalizer: Optional[TextNormallizer] | Callable[[str], str] | bool = None):
        """
        Инициализация Embedder.

        Args:
            normalizer (TextNormallizer | bool | Callable | None) Text normalizer. 
                If None or False, no normalization for text is needed. If Callable
                If True standard mirage.embedders.TextNormalizer is applied
                Any TextNormalizer inherited object is allowed
                Any function: str -> str that normalize text is allowed
        """
        self.normalizer = normalizer

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
        else:
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
    
    @abstractmethod
    def process_chunks(self, chunks: ChunkStorage) -> Dict[int, np.ndarray]:
        """
        Абстрактный метод для обработки чанков и преобразования их в словарь векторов.

        Args:
            chunks (ChunkStorage): Хранилище чанков, которые нужно обработать.

        Returns:
            Dict[int, np.ndarray]: Словарь, где ключ — идентификатор чанка, а значение — векторное представление чанка.
        """
        raise NotImplementedError  