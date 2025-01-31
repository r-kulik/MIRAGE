

from numpy import ndarray
from typing import Callable, Optional
from mirage.embedders import Embedder, TextNormalizer
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder(Embedder):
    def __init__(self, model_name: str = "sentence-transformers/distilbert-base-nli-stsb-quora-ranking", normalizer: Optional[TextNormalizer] | Callable[[str], str] | bool = None):
        """
        Инициализация HuggingFaceEmbedder.

        Args:
            model_name (str): Название модели из библиотеки Hugging Face Sentence Transformers.
            normalizer (TextNormalizer | bool | Callable | None): Нормализатор текста.
        """
        super().__init__(normalizer)
        self.model = SentenceTransformer(model_name)
        self._dim = self.model.get_sentence_embedding_dimension()  # Устанавливаем размерность вектора

    def embed(self, text: str) -> ndarray:
        """
        Векторизация текста с использованием модели Hugging Face Sentence Transformer.

        Args:
            text (str): Входной текст.

        Returns:
            np.ndarray: Векторное представление текста в виде массива numpy.
        """
        # Нормализация текста, если задан нормализатор
        normalized_text = self._normalize(text) 
        # Векторизация текста с помощью модели SentenceTransformer
        vector = self.model.encode(normalized_text)
        
        return vector
    
    def fit(*args, **kwargs):
        pass
    