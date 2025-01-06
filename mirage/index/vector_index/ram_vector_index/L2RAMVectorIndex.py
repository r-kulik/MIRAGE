from .RAMVectorIndex import RAMVectorIndex
from ..VectorIndex import QueryResult, VectorKeyPair, VectorIndex
from typing import Self, Generator
import numpy as np

# TODO: Если почесать репу, то можно представить вектора в индексе как одну большую матрицу, и при вычислении расстояния 
#       от вектора запроса до всех остальных векторов в индексе использовать матричные операции
#       которые вроде как должны быть быстрее цикла for по векторам. Явялется ли это нарушением измерения baseline?


class L2RAMVectorIndex(RAMVectorIndex):
    """Index based on a euclidian distance between vectors with no hierarchy presented. Can ne used as a general truth for performance measuring of other distances

    Parameters
    ----------
    RAMVectorIndex 
        Parent class
    """

    def __init__(self, dimensionality) -> Self:
        super().__init__(dimensionality)
        self.vector_pairs: list[VectorKeyPair] = []

    def _recreate_index(self, vectors: list[np.ndarray]):
        self.vector_pairs = []
        for vector in vectors:
            self.vector_pairs.append(
                VectorKeyPair(
                    vector=vector,
                    chunk_storage_key=None
                )
            )

    def __contains__(self, vector: np.ndarray) -> bool:
        """
        Performs 
        ```py
        vector in L2RAMVectorIndex()
        ``` 
        functionality
        
        Args:
            vector: numpy.ndarray - a vector we are trying to find in the index

        Returns:
            True if there is a vector in index, False overwise
        """
        for vector_key_pair in self.vector_pairs:
            if all(vector_key_pair.vector == vector):
                return True
        return False
    
    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        """
        Performs 
        ```py
        for vector, key in L2RAmVectorIndexObject:
            type(vector) == np.ndarray # True
            type(key) == str # True
            
        ```
        or
        ```
        for vectorKeyPair in L2RAMVectorIndexObject:
            type(vectorKeyPair) == vector_index.VectorKeyPair # True

        ```
        """
        return (
            vkp for vkp in self.vector_pairs
        )

    def add(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        """
        Adds vector to the L2RAMVectorIndex

        Args:
            vector: ndarray - value of vector to store
            chunk_storage_key: str - index of chunk belonging to this vector in the ChunkStorage
        """
        if vector.shape[0] != self.dim:
            raise ValueError(
                f"Vector shape = {vector.shape} does not equal to the dimensionality of VectorIndex {self.dim}"
            )
        self.vector_pairs.append(
            VectorKeyPair(
                vector=vector,
                chunk_storage_key=chunk_storage_key
            )
        )

    
    def attach_chunk_storage_key_to_vector(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        for vkp in self.vector_pairs:
            if all(vkp.vector ==  vector):
                vkp.chunk_storage_key = chunk_storage_key
                break
        else:
            raise VectorIndex.VectorIsNotPresentedInTheIndexException(vector)   

    
    def remove(self, vector: np.ndarray):
        for i in range(len(self.vector_pairs)):
            if all(self.vector_pairs[i].vector == vector):
                self.vector_pairs.pop(i)
                break
        else:
            raise VectorIndex.VectorIsNotPresentedInTheIndexException(vector)
    

    def query(self, query_vector: np.ndarray, top_k = 5) -> list[QueryResult]:
        return sorted(
            [
                QueryResult(
                    distance = np.sqrt(np.sum((query_vector - vkp.vector) ** 2)),
                    # distance = np.dot(query_vector, vkp.vector) / (np.linalg.norm(query_vector) * np.linalg.norm(vkp.vector)),
                    vector=vkp.vector,
                    chunk_storage_key=vkp.chunk_storage_key
                )
                for vkp in self.vector_pairs
            ],
            key=lambda x: x.distance
        )[  :min(len(self.vector_pairs),  top_k)]
    
    
