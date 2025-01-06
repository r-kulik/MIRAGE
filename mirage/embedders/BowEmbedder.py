import numpy as np
from sklearn.feature_extraction.text import CountVectorizer

from mirage.index.vector_index import VectorIndex
from .Embedder import Embedder, EmbedderIsNotTrainedException
from ..index.chunk_storages.ChunkStorage import ChunkStorage
from .TextNormalizer import TextNormalizer
from typing import Dict, Optional
import logging



class BowEmbedder(Embedder):
    """
    A vectorizator that uses Bag of Words method to create a text vector
    
    """

    def __init__(self, normalizer: Optional[TextNormalizer] = None):
        """
        Инициализация BowEmbedder.

        Args:
            normalizer (TextNormallizer | None): Нормализатор текста. Если None, нормализация не применяется.
        """
        super().__init__(normalizer)
        # print(self._dim)
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

        self._dim = len(self.vocabulary) # setting the dimensionality of an embedder equal to the size of vocabulary
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
            raise EmbedderIsNotTrainedException
        
        normalized_text = self._normalize(text)  # Нормализация текста
        vector = self.vectorizer.transform([normalized_text]).toarray()[0]  # Преобразование в вектор
        return vector


    '''   

    DEPRECATED:

    В самом коде конкретной реализации векторизаторов нужно избегать переопределения высокоуровневых функций абстрактного векторизатора связанных с взаимодействием
    с другими компонентами библиотеки MIRAGE
    Но на всякий случай я пока этот код оставлю

    def process_chunks(self, chunks: ChunkStorage):
        """
        Обработка чанков и преобразование их в словарь векторов используя метод Bag of Words
        Если модель не обучена, вызывается метод fit для обучения на переданных чанках.

        Args:
            chunks (ChunkStorage): Хранилище чанков.

        Returns:
            Dict[int, np.ndarray]: Словарь, где ключ — идентификатор чанка, а значение — векторное представление чанка.
        """
        if not self.is_fitted:
            # FIX: Avoid using the print instructions in the source code of MIRAGE. Instead use logging package
            logging.info("Модель не обучена. Выполняется обучение на переданных чанках...")
            self.fit(chunks)  # Обучаем модель на переданных чанках
        return super().process_chunks(chunks)
    
    def convert_chunks_to_vector_index(self, 
                                        chunk_storage: ChunkStorage,
                                        vector_index_object: VectorIndex,
                                        visualize: bool = False
                                        ) -> None:
        if not self.is_fitted:
            # FIX: Avoid using the print instructions in the source code of MIRAGE. Instea use logging package
            if visualize: print(logging.info("Модель не обучена. Выполняется обучение на переданных чанках..."))
            self.fit(chunk_storage)  # Обучаем модель на переданных чанках
        return super().convert_chunks_to_vector_index(chunk_storage, vector_index_object, visualize)

    '''