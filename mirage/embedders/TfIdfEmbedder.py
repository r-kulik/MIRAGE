from numpy import ndarray
from .Embedder import Embedder, EmbedderIsNotTrainedException
from sklearn.feature_extraction.text import TfidfVectorizer
from ..index.chunk_storages.ChunkStorage import ChunkStorage



class TfIdfEmbedder(Embedder):
    """
    Class that implements the TF-IDF vectorization of the text

    Unique methods
    -------------
        `fit(chunks: ChunkStorage)`
        `embed(text: str) -> numpy.ndarray`

    """

    def __init__(self, normalizer = None):
        super().__init__(normalizer)
        self.vectorizer = TfidfVectorizer()
        self.is_fitted = False  # in the moment of initialization TfIdfEmbedder need to be trained on a corpora
        self._dim = -1  # dimensionality of the vectorizator is not known before training

    def fit(self, chunks: ChunkStorage):
        """Training the TF-IDF Embedder on the corpora stored in the presented ChunkStorage

        Parameters
        ----------
        chunks : `ChunkStorage`
            ChunkStorage object to take the chunks with the text from
        """
        train_sentences = [
            self._normalize(chunk_text) for _, chunk_text in chunks
        ]
        self.vectorizer.fit(train_sentences)
        self.vocabulary = self.vectorizer.vocabulary_
        self._dim = len(self.vocabulary)
        self.is_fitted = True
        
    def embed(self, text: str) -> ndarray:
        """Function that converts text to a vector using Tf-Idf vectorization algorithm

        Parameters
        ----------
        text : str
            text that you want to be converted to the vector

        Returns
        -------
        ndarray
            vector of the text

        Raises
        ------
        `EmbedderIsNotTrainedException`
            If embedder was not trained before calling this method. Try `TfIdfEmbedder().fit(ChunkStorage())` before using it
        """
        if not self.is_fitted:
            raise EmbedderIsNotTrainedException

        normalized_text = self._normalize(text)
        vector = self.vectorizer.transform([normalized_text]).toarray()[0]
        return vector