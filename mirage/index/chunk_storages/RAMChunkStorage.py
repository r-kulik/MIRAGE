from .ChunkStorage import ChunkStorage


class RAMChunkStorage(ChunkStorage):
    def __init__():
        super().__init__()
    def __getitem__(self, index) -> str:
        return self._chunk_map[index].link_to_chunk
        