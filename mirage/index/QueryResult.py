from typing import Optional, Any, Self
from pydantic import BaseModel


class QueryResult(BaseModel):
    score: float
    chunk_storage_key: str
    vector: Optional[Any] = None

    def __hash__(self):
        return hash(self.chunk_storage_key)

    def __eq__(self, value: Self) -> bool:
        return self.chunk_storage_key == value.chunk_storage_key
