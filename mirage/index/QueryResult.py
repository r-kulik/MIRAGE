from typing import Optional, Any
from pydantic import BaseModel

class QueryResult(BaseModel):
    score: float
    chunk_storage_key: str
    vector: Optional[Any] = None

    def __hash__(self):
        return hash(self.chunk_storage_key)