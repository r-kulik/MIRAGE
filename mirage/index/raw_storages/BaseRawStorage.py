

class BaseRawStorage:
    """
    Every Raw Storage can be represented to the MIRAGE as a dictionary that maps index of the file to its raw text content.
    Keys can be anything: database indexes, urls, file_paths.
    """

    class IndexIsAlreadyInStorageException(Exception):
        def __str__(self): return "Index you are trying to add is already presented in the storage. Avoid using non-unique names"

    def get_indexes(self) -> list[str]:
        pass

    def addToStorage(self, index: str, link: any) -> None:
        pass

    def __getitem__(self, index: str) -> str:
        pass

    def get_link(self, index: str) -> any:
        pass


