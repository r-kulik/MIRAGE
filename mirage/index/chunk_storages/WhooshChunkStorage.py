import uuid
from whoosh import index, fields, scoring, qparser
from whoosh.analysis import StemmingAnalyzer
from whoosh.filedb.filestore import RamStorage
from whoosh.qparser import syntax
from typing import Callable, Generator, List, Optional

from pydantic import BaseModel
from typing import Literal

from ...embedders import TextNormalizer
from . import ChunkStorage


class WhooshChunkStorage(ChunkStorage):

    def __init__(
        self,
        scoring_function: Literal["BM25", "BM25F", "TF-IDF"],
        normalizer: Optional[TextNormalizer] | bool | Callable[[str], str] = True,
        K: Optional[float] = None,
        B1: Optional[float] = None
    ):
        super().__init__(scoring_function)
        self.scoring_function = scoring_function
        self.K = K
        self.B1 = B1
        if type(normalizer) == bool and normalizer:
            normalizer = TextNormalizer(stop_word_remove=True, word_generalization="stem")
        self.normalizer = normalizer

        self._setup_index()

    def __normalize(self, text: str) -> str:
        if self.normalizer:
            return self.normalizer(text)
        return text

    def _setup_index(self):
        schema = fields.Schema(
            id=fields.ID(stored=True, unique=True),
            text=fields.STORED,  # Original text
            normalized_text=fields.TEXT(stored=True),  # Normalized for search
            raw_document_index=fields.STORED
        )
        self.storage = RamStorage()
        self.ix = self.storage.create_index(schema)

    def _get_weighting(self):
        if self.scoring_function == "BM25F":
            kwargs = {}
            if self.K is not None:
                kwargs["K1"] = self.K
            if self.B1 is not None:
                kwargs["B"] = self.B1
            return scoring.BM25F(**kwargs)
        elif self.scoring_function == "BM25":
            return scoring.BM25F()  # Original behavior for BM25
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

    def get_normalized_text(self, index: str) -> str:
        with self.ix.searcher() as searcher:
            doc = searcher.document(id=index)
            if not doc:
                raise KeyError(f"Chunk index {index} not found")
            return doc['normalized_text']

    def add_chunk(self, text: str, raw_document_index: str) -> str:
        chunk_id = str(uuid.uuid4())
        with self.ix.searcher() as searcher:
            if searcher.document(id=chunk_id):
                raise self.ChunkIndexIsAlreadyInStorageException()
        
        normalized_text = self.__normalize(text)  # Assume __normalize is implemented
        writer = self.ix.writer()
        writer.add_document(
            id=chunk_id,
            text=text,
            normalized_text=normalized_text,
            raw_document_index=raw_document_index
        )
        writer.commit()
        return chunk_id

    def clear(self) -> None:
        writer = self.ix.writer()
        writer.delete_by_query(qparser.QueryParser(fieldname="normalized_text", schema=self.ix.schema).parse("*"))
        writer.commit()

    def __iter__(self) -> Generator[tuple[str, str], None, None]:
        with self.ix.searcher() as searcher:
            for doc in searcher.all_stored_fields():
                yield (doc['id'], doc['text'])

    def query(self, query: str) -> List[ChunkStorage.ChunkNote]:
        query = self.__normalize(query)
        with self.ix.searcher(weighting=self._get_weighting()) as searcher:
            parser = qparser.QueryParser("normalized_text", self.ix.schema, group=syntax.OrGroup)
            q = parser.parse(query)
            results = searcher.search(q)
            return [self.ChunkNote(
                text=hit['text'],
                raw_document_index=hit['raw_document_index']
            ) for hit in results]