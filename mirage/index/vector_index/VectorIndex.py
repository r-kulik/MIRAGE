from abc import ABC, abstractmethod
import numpy as np
from typing import Generator, Self
from ..QueryResult import QueryResult

    
class VectorKeyPair:
    def __init__(self, vector: np.ndarray, chunk_storage_key: str) -> Self: 
        self.vector = vector; self.chunk_storage_key = chunk_storage_key

    def __iter__(self):
        """
        Enable unpacking of the object into tuple of (vector, chunk_storage_key).
        ```py
        >>> vector, key = VectorKeyPair(v, k)
        >>> vector == v and key == k
        True
        ```
        """
        yield self.vector
        yield self.chunk_storage_key
    
    def __str__(self) -> str: return f"VectorKeyPair(vector={self.vector}; chunk_storage_key={self.chunk_storage_key})"


class VectorIndex(ABC):
    """
    Base anstract class that provides interfaces for vector storages of chunks
    
    Attributes:
        dim: int - dimensionality of the vector space of index
    """

    def __init__(self, dimensionality: int):
        super().__init__()
        self.dim = dimensionality
        self.is_trained = None

    @abstractmethod
    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        "This function shall return pairs of (vector: ndarray, chunk_storage_key: str)"
        raise NotImplementedError
    
    @abstractmethod
    def __contains__(self, vector: np.ndarray) -> bool:
        """
        This function implements
        ```py
        vector in VectorIndex 
        ```
        functionality
        """
        raise NotImplementedError

    @abstractmethod
    def add(self, vector: np.ndarray, chunk_storage_key: int) -> None:
        """
        Function to add a new entry to the storage

        Args:
            vector: np.ndarray with the same dimenstionality as self.dim
            chunk_storage_index: int - id of the chink that belongs to this vector
        """
        raise NotImplementedError
    
    @abstractmethod
    def query(self, query_vector: np.ndarray, top_k: int = 1) -> list[QueryResult]:
        """
        Function for quiering the vector index. Rerutns top_k vectors closest (according to the structure of index) to the quiery_vector
        """
        raise NotImplementedError
    

    class VectorIsNotPresentedInTheIndexException(Exception):
        '''
        This exception occurs when where are manipulations supposed to be executed on a vector that is not presented in the index
        '''
        def __init__(self, vector):
            super().__init__(
                f"You are trying to delete or edit vector {vector} that is not presented in the index"
            )

    @abstractmethod
    def attach_chunk_storage_key_to_vector(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        """
        This function allows to redefine the chunk belonging to vector. It is needed to restore the index from the file, and attach chiks to vectors after the hierarchy is restored
        ```py

        >>> index = VectorIndex()

        >>> index.add(
                vector=np.array([1]), 
                chunk_storage_key='a'
            )

        >>> index.query(np.array([1]))
        QueryResult(distance = 0, vector = array([1]), chunk_storage_index = 'a')

        >>> index.attach_chunk_storage_key_to_vector(
                vector=np.array([1]),
                chunk_storage_key = 'b'
            )

        >>> index.query(np.array([1]))
        QueryResult(distance = 0; vector = array([1]), chunk_storage_index = 'b')
        ```
        """
        raise NotImplementedError
    
    @abstractmethod
    def train(self) -> None:
        pass


    