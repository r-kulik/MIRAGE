from abc import abstractmethod, ABC
import typing
from typing import Callable, Generator, List, Literal
from pydantic import BaseModel

from mirage.index import QueryResult


class ChunkNote(BaseModel):
    text: str
    raw_document_index: str

    def __hash__(self):
        return (hash(self.text) + hash(self.raw_document_index)) // 2


class ChunkStorage(ABC):
    """This class defines the interface of the storing and querying with the full-text search the storage of text chunks

    Raises
    ------
    ChunkStorage.ChunkIndexIsAlreadyInStorageException
        Raises when a text chunk which is already persented in the storage is trying to be added
    """

    class ChunkIndexIsAlreadyInStorageException(Exception):
        def __str__(self):
            return "You are trying to add in storage a chunk which index is already presented in a storage"

    def __init__(self, scoring_function: Literal["BM25", "BM25F", "TF-IDF"]):
        """In initialization, the ChunkStorage is empty, and no assumptions about inner structure of the storage is not done.
        But scoring function to fulltext search is defined by Literal, or by function.
        The scoring function is redefined after the initalization.
        """
        super().__init__()
        self.scoring_function_name = scoring_function

    @abstractmethod
    def get_indexes(self) -> list[str]:
        """Returns all indexes of chunks that are presented in the storage

        Returns
        -------
        list[str]
            List of available indexes
        """
        pass

    @abstractmethod
    def get_raw_index_of_document(self, index: str) -> str:
        """Each chunk is attached to the document.
        This function allows to get the index of the source document the text chunk was created from

        Parameters
        ----------
        index : str
            Index of chunk in ChunkStorage

        Returns
        -------
        str
            Index of document in the RawStorage
        """
        pass

    @abstractmethod
    def __getitem__(self, index: str) -> str:
        """Getting a text chunk from the storage by its index

        Parameters
        ----------
        index : str
            index of text chunk

        Returns
        -------
        str
            the text from the chunk
        """

    @abstractmethod
    def add_chunk(self, text: str, raw_document_index: str) -> str:
        """Adding the chunk of text in the ChunkStorage with the link to the document index

        Parameters
        ----------
        text : str
            Text of the chunk
        raw_document_index : str
            Index of document in RawStorage this text was originated from

        Returns
        -------
        str
            Index of the text chunk that is added
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Сlearing the whole storage, by deleting all the chunks"""
        pass

    def __iter__(self) -> Generator[tuple[str], None, None]:
        """
        Returns generator of the following:
        ```
        >>> for chunk_index, chunk_text in ChunkStorageObject:
                type(chunk_index) == str # True
                type(chunk_text) == str # True
        ```
        """
        pass

    @abstractmethod
    def query(self, query: str) -> list[QueryResult]:
        """The query of full-text search among the chunk storage

        Parameters
        ----------
        query : str
            User query

        Returns
        -------
        list[ChunkNote]
            The list objects containing fields:
                .text - text of the chunk
                .raw_document_index - the index of document in RawStorage this chunk was originated from
        """
        pass

    def get_texts_for_search_results(
        self, search_results: List[QueryResult]
    ) -> List[str]:
        return [self.__getitem__(result.chunk_storage_key) for result in search_results]
