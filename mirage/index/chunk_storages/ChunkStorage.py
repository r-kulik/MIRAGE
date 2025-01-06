from abc import abstractmethod, ABC
import typing
from typing import Generator

class ChunkStorage(ABC):

    class _ChunkNote:

        def __init__(self, link_to_chunk: str, raw_index_of_document: str) -> None:
            self.link_to_chunk: str = link_to_chunk
            self.raw_index_of_document: str = raw_index_of_document

        def __hash__(self) -> int:
            return (self.link_to_chunk, self.raw_index_of_document).__hash__()
        
        def __eq__(self, value: object) -> bool:
            return self.link_to_chunk == value.link_to_chunk and self.raw_index_of_document == value.raw_index_of_document
        
    class ChunkIndexIsAlreadyInStorageException(Exception):
        def __str__(self): return "You are trying to add in storage a chunk which index is already presented in a storage"

    def __init__(self):
        self._chunk_map: dict[str, ChunkStorage._ChunkNote] = {}

    def _addToChunkIndex(self, index: str, link_to_chunk: typing.Any, raw_index_of_document: str) -> None:
        if index in self._chunk_map:
            raise ChunkStorage.ChunkIndexIsAlreadyInStorageException
        self._chunk_map[index] = ChunkStorage._ChunkNote(link_to_chunk, raw_index_of_document)

    def get_indexes(self) -> list[str]:
        return list(self._chunk_map.keys())

    def get_raw_index_of_document(self, index: str) -> str:
        return self._chunk_map[index].raw_index_of_document


    def __getitem__(self, index: str) -> str:
        raise NotImplementedError("Subclasses must implement this functionality")
    
    @abstractmethod
    def add_chunk(self, text: str, raw_index_of_document: str) -> str:
        raise NotImplementedError("Subclasses must implement this functionality")
    
    def clear(self) -> None:
        raise NotImplementedError("Subclasses must implement this functionality")
    
    def __iter__(self) -> Generator[tuple[str], None, None]:
        """
        Returns generator of the following:
        ```py
        >>> for chunk_index, chunk_text in ChunkStorageObject:
                type(chunk_index) == str # True
                type(chunk_text) == str # True
        ```
        """     
        raise NotImplementedError("Subclasses must implement this functionality")
