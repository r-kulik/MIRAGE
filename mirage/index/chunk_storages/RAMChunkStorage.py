from .ChunkStorage import ChunkStorage


class RAMChunkStorage(ChunkStorage):
    def __init__(self):
        super().__init__()   
        
    def __getitem__(self, index) -> str:
        return self._chunk_map[index].link_to_chunk
    
    def add_chunk(self, text: str, raw_document_index: str, __offset = "") -> None:
        index = str(hash(text))
        # print(index, text)
        try:
            self._addToChunkIndex(index + __offset, text, raw_document_index)
        except ChunkStorage.ChunkIndexIsAlreadyInStorageException:
            self.add_chunk(text, raw_document_index, __offset + "_hit")
        return index + __offset

    def clear(self) -> None:
        self._chunk_map = {}
        
    def __iter__(self):
        return ((index, self._chunk_map[index].link_to_chunk) for index in self._chunk_map.keys())