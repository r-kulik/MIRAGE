import gc
import json
import os
import shutil
import tempfile
import uuid
import zipfile
from whoosh import index, fields, scoring, qparser
from whoosh.analysis import StemmingAnalyzer
from whoosh.filedb.filestore import RamStorage, FileStorage, copy_to_ram
from whoosh.qparser import syntax
from typing import Callable, Generator, List, Optional

from pydantic import BaseModel
from typing import Literal

from mirage.index.QueryResult import QueryResult

from ...embedders import TextNormalizer
from .ChunkStorage import ChunkNote, ChunkStorage


class WhooshChunkStorage(ChunkStorage):

    def __init__(
        self,
        scoring_function: Literal["BM25", "BM25F", "TF-IDF"],
        normalizer: Optional[TextNormalizer] | bool | Callable[[str], str] = True,
        K1: Optional[float] = None,
        B: Optional[float] = None,
    ):
        super().__init__(scoring_function)
        self.scoring_function = scoring_function
        self.K1 = K1
        self.B = B
        if type(normalizer) == bool and normalizer:
            normalizer = TextNormalizer(
                stop_word_remove=True, word_generalization="stem"
            )
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
            raw_document_index=fields.STORED,
        )
        self.storage = RamStorage()
        self.ix = self.storage.create_index(schema)

    def _get_weighting(self):
        if self.scoring_function == "BM25F":
            kwargs = {}
            if self.K1 is not None:
                kwargs["K1"] = self.K1
            if self.B is not None:
                kwargs["B"] = self.B
            return scoring.BM25F(**kwargs)
        elif self.scoring_function == "BM25":
            return scoring.BM25F()  # Original behavior for BM25
        elif self.scoring_function == "TF-IDF":
            return scoring.TF_IDF()
        else:
            raise ValueError(f"Unsupported scoring function: {self.scoring_function}")

    def get_indexes(self) -> List[str]:
        with self.ix.searcher() as searcher:
            return [doc["id"] for doc in searcher.all_stored_fields()]

    def get_raw_index_of_document(self, index: str) -> str:
        with self.ix.searcher() as searcher:
            doc = searcher.document(id=index)
            if not doc:
                raise KeyError(f"Chunk index {index} not found")
            return doc["raw_document_index"]

    def __getitem__(self, index: str) -> str:
        with self.ix.searcher() as searcher:
            doc = searcher.document(id=index)
            if not doc:
                raise KeyError(f"Chunk index {index} not found")
            return doc["text"]

    def get_normalized_text(self, index: str) -> str:
        with self.ix.searcher() as searcher:
            doc = searcher.document(id=index)
            if not doc:
                raise KeyError(f"Chunk index {index} not found")
            return doc["normalized_text"]

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
            raw_document_index=raw_document_index,
        )
        writer.commit()
        return chunk_id

    def clear(self) -> None:
        writer = self.ix.writer()
        writer.delete_by_query(
            qparser.QueryParser(
                fieldname="normalized_text", schema=self.ix.schema
            ).parse("*")
        )
        writer.commit()

    def __iter__(self) -> Generator[tuple[str, str], None, None]:
        with self.ix.searcher() as searcher:
            for doc in searcher.all_stored_fields():
                yield (doc["id"], doc["text"])

    def query(self, query: str) -> List[QueryResult]:
        query = self.__normalize(query)
        with self.ix.searcher(weighting=self._get_weighting()) as searcher:
            parser = qparser.QueryParser(
                "normalized_text", self.ix.schema, group=syntax.OrGroup
            )
            q = parser.parse(query)
            results = searcher.search(q)
            return [
                QueryResult(score=hit.score, chunk_storage_key=hit["id"], vector=None)
                for hit in results
            ]

    def save(self, path: str) -> None:
        """Сохраняет индекс и настройки в zip-файл с расширением .whoosh."""
        if not path.endswith(".whoosh"):
            raise ValueError("Файл должен иметь расширение .whoosh")

        # Создаем временную директорию
        with tempfile.TemporaryDirectory() as tmpdirname:
            temp_storage = FileStorage(tmpdirname)
            temp_index = temp_storage.create_index(self.ix.schema)

            # Копируем все документы
            writer = temp_index.writer()
            with self.ix.searcher() as searcher:
                for fields in searcher.all_stored_fields():
                    writer.add_document(**fields)
            writer.commit()

            # Сохраняем метаданные
            metadata = {
                "scoring_function": self.scoring_function,
                "K1": self.K1,
                "B": self.B,
                "normalizer": bool(
                    self.normalizer
                ),  # Сохраняем только факт использования нормализатора
            }
            with open(
                os.path.join(tmpdirname, "metadata.json"), "w", encoding="utf-8"
            ) as f:
                json.dump(metadata, f)

            # Упаковываем директорию в zip-архив с расширением .whoosh
            with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(tmpdirname):
                    for file in files:
                        filepath = os.path.join(root, file)
                        arcname = os.path.relpath(filepath, tmpdirname)
                        zf.write(filepath, arcname)

    @classmethod
    def load(cls, path: str) -> "WhooshChunkStorage":
        """
        Загружает индекс и настройки из zip-архива .whoosh и возвращает
        полностью функциональный экземпляр WhooshChunkStorage с RamStorage.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(f"Файл {path} не найден")

        # распакуем в «temporary directory» — контекстный менеджер сам позаботится об удалении
        with tempfile.TemporaryDirectory() as tmpdir:
            with zipfile.ZipFile(path, "r") as zf:
                zf.extractall(tmpdir)

            # прочитаем метаданные
            meta_path = os.path.join(tmpdir, "metadata.json")
            if not os.path.isfile(meta_path):
                raise ValueError("В архиве нет файла metadata.json")
            with open(meta_path, "r", encoding="utf-8") as mf:
                meta = json.load(mf)

            scoring_function = meta["scoring_function"]
            K1 = meta.get("K1")
            B = meta.get("B")
            norm_flag = meta.get("normalizer", True)

            # создаём экземпляр с нужными параметрами
            instance = cls(
                scoring_function=scoring_function, normalizer=norm_flag, K1=K1, B=B
            )

            # открываем файловое хранилище и сразу копируем его в память
            file_storage = FileStorage(tmpdir)
            ram_storage = copy_to_ram(
                file_storage
            )  # копирует все файлы в RamStorage :contentReference[oaicite:0]{index=0}

            # создаём индекс поверх RamStorage
            instance.storage = ram_storage
            instance.ix = ram_storage.open_index()

            # к этому моменту ни один дескриптор к файлам на диске не держится,
            # и TemporaryDirectory при выходе спокойно удалит tmpdir

        return instance
