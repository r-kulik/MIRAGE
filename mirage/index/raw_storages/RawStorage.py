

from typing import Self


class RawStorage:
    """
    Every Raw Storage can be represented to the MIRAGE as a dictionary that maps index of the file to its raw text content.
    Keys can be anything: database indexes, urls, file_paths.
    """

    class IndexIsAlreadyInStorageException(Exception):
        def __str__(self): return "Index you are trying to add is already presented in the storage. Avoid using non-unique names"

    def __init__(self):
        self._storage: dict[str, str] = {}
    
    def get_indexes(self) -> list[str]:
        return list(self._storage.keys())

    def add_to_storage(self, index: str, link: str) -> None:
        if index not in self._storage:
            self._storage[index] = link
        else:
            raise RawStorage.IndexIsAlreadyInStorageException

    def __getitem__(self, index: str) -> str:
        pass

    def get_link(self, index: str) -> any:
        return self._storage[index]
    
    def save(self, filename: str) -> None:
        raise NotImplementedError("Concrete implementations of the RawStorage must handle save and load functionality")
    
    @staticmethod
    def load(filename) -> Self:
        raise NotImplementedError("Concrete implementations of the RawStorage must handle save and load functionality")

