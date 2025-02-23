import uuid
from whoosh import index, fields, scoring, qparser
from whoosh.analysis import StemmingAnalyzer
from whoosh.filedb.filestore import RamStorage
from typing import Generator, List

from pydantic import BaseModel
from typing import Literal
from . import ChunkStorage

class WhooshChunkStorage(ChunkStorage):

    def __init__(self, scoring_function: Literal["BM25", "BM25F", "TF-IDF"]):
        super().__init__(scoring_function)
        self.scoring_function = scoring_function
        self._setup_index()

    def _setup_index(self):
        schema = fields.Schema(
            id=fields.ID(stored=True, unique=True),
            text=fields.TEXT(stored=True, analyzer=StemmingAnalyzer()),
            raw_document_index=fields.STORED
        )
        self.storage = RamStorage()
        self.ix = self.storage.create_index(schema)

    def _get_weighting(self):
        if self.scoring_function in ("BM25", "BM25F"):
            return scoring.BM25F()
        elif self.scoring_function == "TF-IDF":
            return scoring.TF_IDF()
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_function}")

    def get_indexes(self) -> List[str]:
        with self.ix.searcher() as searcher:
            return [doc['id'] for doc in searcher.all_stored_fields()]

    def get_raw_index_of_document(self, index: str) -> str:
        with self.ix.searcher() as searcher:
            doc = searcher.document(id=index)
            if not doc:
                raise KeyError(f"Chunk index {index} not found")
            return doc['raw_document_index']

    def __getitem__(self, index: str) -> str:
        with self.ix.searcher() as searcher:
            doc = searcher.document(id=index)
            if not doc:
                raise KeyError(f"Chunk index {index} not found")
            return doc['text']

    def add_chunk(self, text: str, raw_document_index: str) -> str:
        chunk_id = str(uuid.uuid4())
        with self.ix.searcher() as searcher:
            if searcher.document(id=chunk_id):
                raise self.ChunkIndexIsAlreadyInStorageException()
        writer = self.ix.writer()
        writer.add_document(id=chunk_id, text=text, raw_document_index=raw_document_index)
        writer.commit()
        return chunk_id

    def clear(self) -> None:
        writer = self.ix.writer()
        writer.delete_by_query(qparser.QueryParser(fieldname="text", schema=self.ix.schema).parse("*"))
        writer.commit()

    def __iter__(self) -> Generator[tuple[str, str], None, None]:
        with self.ix.searcher() as searcher:
            for doc in searcher.all_stored_fields():
                yield (doc['id'], doc['text'])

    def query(self, query: str) -> List[ChunkStorage.ChunkNote]:
        with self.ix.searcher(weighting=self._get_weighting()) as searcher:
            parser = qparser.QueryParser("text", self.ix.schema)
            q = parser.parse(query)
            results = searcher.search(q)
            return [self.ChunkNote(text=hit['text'], raw_document_index=hit['raw_document_index']) for hit in results]