from ..VectorIndex import VectorIndex
import os
from abc import ABC, abstractmethod
from numpy import ndarray, array
import json
from typing import final


class RAMVectorIndex(VectorIndex, ABC):

    """
    Base class for all VectorIndexes that are stored in Random Access Memory and were designed by us.
    It provides functions of saving and loading the index into physical memory to enable the inference process without retraining the index

    Saving the Vector Index must also require saving the ChunkStorage and RawStorage to restore the indexes
    """


    def __init__(self, dimensionality: int):
        super().__init__(dimensionality)
    
    @abstractmethod
    def _recreate_index(self, vectors: list[ndarray]):
        """
        This function shall recreate the index given the list of vectors and create hierarchy back and store it into the main memory
        """
        raise NotImplementedError
    
    @abstractmethod
    def attach_chunk_storage_key_to_vector(self, vector: ndarray, chunk_storage_key: str) -> None:
        raise NotImplementedError

    @final
    def save(self, filename: str) -> None:
        """Saves the given RAM Index to the physical memory with the possibility to restore

        Args:
            filename (str): filename of the .json file with the index to be stored
        """
        json_to_save = []
        for vector, chunk_storage_key in self:
            json_to_save.append(
                vector.tolist(), chunk_storage_key
            )
        with open(filename, encoding='utf-8') as file:
            file.write(
                json.dumps(
                    json_to_save
                )
            )
    
    @final
    def load(self, filename: str) -> None:
        """Loads Vector index from the file (.json formatted)

        Parameters
        ----------
        filename : str
            Name of the file to load the index from

        Examples
        ----------

        >>> index1.save('a.json')
        >>> index2 = RAMVectorIndex()
        >>> index2.load('a.json') # index1 and index2 behave the same
        
        """
        with open(filename, encoding='utf-8') as file:
            json_to_load = json.loads(
                file.read()
            )
        self.__recreate_index(
            [
                array(vector) for vector, chunk_storage_key in json_to_load
            ]
        )
        for vector, chunk_storage_key in json_to_load:
            self.attach_chunk_storage_key_to_vector(
                array(vector), chunk_storage_key
            )
        
