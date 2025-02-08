

from numpy import ndarray
from typing import Callable, Optional, Literal
from mirage.embedders import Embedder, TextNormalizer
from sentence_transformers import SentenceTransformer


class HuggingFaceEmbedder(Embedder):
    def __init__(self,
                 model_name: Literal[
                     "sentence-transformers/distilbert-base-nli-stsb-quora-ranking",
                     "Alibaba-NLP/gte-large-en-v1.5",
                     'ai-forever/ruRoberta-large',
                     'DeepPavlov/rubert-base-cased-sentence'
                     ] = "Alibaba-NLP/gte-large-en-v1.5",
                 normalizer: Optional[TextNormalizer] | Callable[[str], str] | bool = None):
        """
        Initialization of HuggingFaceEmbedder.

        Args:
            model_name (str): Название модели из библиотеки Hugging Face Sentence Transformers.
            normalizer (TextNormalizer | bool | Callable | None): Нормализатор текста.
        """
        super().__init__(normalizer)
        self.model = SentenceTransformer(model_name, trust_remote_code=True)
        self._dim = self.model.get_sentence_embedding_dimension()  # Устанавливаем размерность вектора

    def embed(self, text: str) -> ndarray:
        """
        Text Vectorization using Hugging Face Sentence Transformer.

        Args:
            text (str): Input text

        Returns:
            np.ndarray: Vector representation as a numpy ndarray
        """
        # Нормализация текста, если задан нормализатор
        normalized_text = self._normalize(text) 
        # Векторизация текста с помощью модели SentenceTransformer
        vector = self.model.encode(normalized_text)
        
        return vector
    
    def fit(*args, **kwargs):
        pass
    