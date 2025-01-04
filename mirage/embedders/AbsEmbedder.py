from abc import ABC, abstractmethod
from typing import Optional, Dict
import numpy as np
from .TextNormallizer import TextNormallizer
from ..index.chunk_storages import ChunkStorage

class AbsEmbedder(ABC):
    def __init__(self, normalizer: Optional[TextNormallizer] = None):
        """
        Инициализация Embedder.

        Args:
            normalizer (TextNormallizer | None): Нормализатор текста. Если None, нормализация не применяется.
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
        pass
    
    @abstractmethod
    def process_chunks(self, chunks: ChunkStorage) -> Dict[int, np.ndarray]:
        """
        Абстрактный метод для обработки чанков и преобразования их в словарь векторов.

        Args:
            chunks (ChunkStorage): Хранилище чанков, которые нужно обработать.

        Returns:
            Dict[int, np.ndarray]: Словарь, где ключ — идентификатор чанка, а значение — векторное представление чанка.
        """
        pass