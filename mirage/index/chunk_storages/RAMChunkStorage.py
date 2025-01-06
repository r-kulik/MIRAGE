from .ChunkStorage import ChunkStorage


class RAMChunkStorage(ChunkStorage):
    def __init__(self):
        super().__init__()   
        
    def __getitem__(self, index) -> str:
        return self._chunk_map[index].link_to_chunk
    
    def add_chunk(self, text: str, raw_index_of_document: str) -> None:
        index = str(hash(text))
        print(index, text)
        self._addToChunkIndex(index, text, raw_index_of_document)
        return index

    def clear(self) -> None:
        self._chunk_map = {}
        
    def __iter__(self):
        return ((index, self._chunk_map[index].link_to_chunk) for index in self._chunk_map.keys())