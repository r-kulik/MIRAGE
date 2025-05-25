from abc import ABC, abstractmethod
from typing import List, Optional

from mirage.index.chunk_storages import ChunkStorage


class LLMAdapter(ABC):
    def __init__(self):
        pass

    @abstractmethod
    def do_request(
        self,
        query: str,
        chunk_storage: ChunkStorage,
        indexes: List[str],
        prompt_template: Optional[str] = None,
    ): ...
