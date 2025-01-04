from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from .AbsEmbedder import AbsEmbedder
from ..index.chunk_storages.ChunkStorage import ChunkStorage
from .TextNormallizer import TextNormallizer
from typing import Dict, Optional

class BowEmbedder(AbsEmbedder):
    def __init__(self, normalizer: Optional[TextNormallizer] = None):
        """
        Инициализация BowEmbedder.

        Args:
            normalizer (TextNormallizer | None): Нормализатор текста. Если None, нормализация не применяется.
        """
        super().__init__(normalizer)
        self.vectorizer = CountVectorizer()  # Используем CountVectorizer для создания BoW
        self.vocabulary = None  # Словарь уникальных слов
        self.is_fitted = False  # Флаг, указывающий, была ли модель обучена

    def fit(self, chunks: ChunkStorage):
        """
        Обучение модели на всех текстах из чанков для создания словаря.

        Args:
            chunks (ChunkStorage): Хранилище чанков, из которых извлекаются тексты для обучения.
        """
        # Извлекаем все тексты из чанков
        texts = [chunk for _, chunk in chunks]  
        # Нормализация текстов
        normalized_texts = [self._normalize(text) for text in texts]
        
        # Создание словаря
        self.vectorizer.fit(normalized_texts)
        self.vocabulary = self.vectorizer.vocabulary_  # Сохраняем словарь
        self.is_fitted = True  # Устанавливаем флаг, что модель обучена

    def embed(self, text: str) -> np.ndarray:
        """
        Преобразование текста в векторное представление с использованием BoW.

        Args:
            text (str): Входной текст.

        Returns:
            np.ndarray: Векторное представление текста.
        """
        if not self.is_fitted:
            raise ValueError("Модель не обучена. Сначала вызовите метод fit.")
        
        normalized_text = self._normalize(text)  # Нормализация текста
        vector = self.vectorizer.transform([normalized_text]).toarray()[0]  # Преобразование в вектор
        return vector

    def process_chunks(self, chunks: ChunkStorage) -> Dict[int, np.ndarray]:
        """
        Обработка чанков и преобразование их в словарь векторов.
        Если модель не обучена, вызывается метод fit для обучения на переданных чанках.

        Args:
            chunks (ChunkStorage): Хранилище чанков.

        Returns:
            Dict[int, np.ndarray]: Словарь, где ключ — идентификатор чанка, а значение — векторное представление чанка.
        """
        if not self.is_fitted:
            print("Модель не обучена. Выполняется обучение на переданных чанках...")
            self.fit(chunks)  # Обучаем модель на переданных чанках

        vectors = {}
        for chunk_id, chunk in chunks:  # Предполагаем, что ChunkStorage поддерживает метод items()
            vector = self.embed(chunk)  # Преобразуем каждый чанк в вектор
            vectors[chunk_id] = vector
        return vectors