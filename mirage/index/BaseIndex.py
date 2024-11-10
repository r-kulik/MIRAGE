from mirage.index.raw_storages import BaseRawStorage


class BaseIndex:
    def __init__(self, raw_storage: BaseRawStorage, chunker: BaseChunker) -> None:
        pass