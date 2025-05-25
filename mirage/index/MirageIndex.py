import importlib
import json
import os
import pickle
import tempfile
import zipfile
import zlib
from mirage.index.chunk_storages import ChunkStorage
from mirage.index.chunking_algorithms import ChunkingAlgorithm
from mirage.index.raw_storages import RawStorage
from mirage.index.vector_index.VectorIndex import VectorIndex, QueryResult
from ..embedders import Embedder
from abc import abstractmethod, ABC
from typing import Any, Optional, final
from loguru import logger

logger.disable(__name__)


class MirageIndex(ABC):

    def __init__(
        self,
        raw_storage: Optional[RawStorage],
        chunk_storage,
        chunking_algorithm,
        vector_index,
        visualize=False,
    ):
        super().__init__()
        self.raw_storage: RawStorage = raw_storage
        self.chunk_storage: ChunkStorage = chunk_storage
        self.chunking_algorithm: ChunkingAlgorithm = chunking_algorithm
        self.vector_index: VectorIndex = vector_index
        self.visualize = visualize

    def save(self, filename_to_save: str) -> None:
        """Сохраняет весь индекс в zip-архив"""

        logger.info(f"Saving Mirage index to {filename_to_save}...")

        with zipfile.ZipFile(filename_to_save, "w", zipfile.ZIP_DEFLATED) as zipf:
            # Сохранение RawStorage
            raw_storage_filename = "raw_storage.mirage_storage"
            self.raw_storage.save(
                raw_storage_filename
            )  # предполагаем, что метод save в RawStorage уже есть
            zipf.write(raw_storage_filename)
            os.remove(raw_storage_filename)

            # Сохранение ChunkStorage
            chunk_storage_filename = "chunk_storage.whoosh"
            self.chunk_storage.save(
                chunk_storage_filename
            )  # предполагаем, что метод save в ChunkStorage есть
            zipf.write(chunk_storage_filename)
            os.remove(chunk_storage_filename)

            # Сохранение VectorIndex
            vector_index_filename = "vector_index.faiss"
            self.vector_index.save(
                vector_index_filename
            )  # предполагаем, что метод save в FaissIndex есть
            zipf.write(vector_index_filename)
            os.remove(vector_index_filename)

            # Добавление конфигурации метаданных
            metadata = {
                "raw_storage": {
                    "module": self.raw_storage.__class__.__module__,
                    "class": self.raw_storage.__class__.__name__,
                },
                "chunk_storage": {
                    "module": self.chunk_storage.__class__.__module__,
                    "class": self.chunk_storage.__class__.__name__,
                },
                "chunking_algorithm": {
                    "module": self.chunking_algorithm.__class__.__module__,
                    "class": self.chunking_algorithm.__class__.__name__,
                },
                "vector_index": {
                    "module": self.vector_index.__class__.__module__,
                    "class": self.vector_index.__class__.__name__,
                },
                "visualize": self.visualize,
                "raw_storage_file": raw_storage_filename,
                "chunk_storage_file": chunk_storage_filename,
                "vector_index_file": vector_index_filename,
            }
            zipf.writestr("metadata.json", json.dumps(metadata, indent=4))

            logger.info(f"Mirage index saved to {filename_to_save}.")

    @staticmethod
    def load(filename: str) -> "MirageIndex":
        """Загружает индекс из zip-архива"""

        logger.info(f"Loading Mirage index from {filename}...")

        with zipfile.ZipFile(filename, "r") as zipf:
            # Извлечение метаданных
            with zipf.open("metadata.json") as metadata_file:
                metadata = json.loads(metadata_file.read())

            # Загрузка RawStorage
            raw_storage_filename = metadata["raw_storage_file"]
            zipf.extract(raw_storage_filename)
            raw_storage_module = importlib.import_module(
                metadata["raw_storage"]["module"]
            )
            raw_storage_class = getattr(
                raw_storage_module, metadata["raw_storage"]["class"]
            )
            raw_storage = raw_storage_class.load(
                raw_storage_filename
            )  # предполагаем, что метод load в RawStorage работает
            os.remove(raw_storage_filename)

            # Загрузка ChunkStorage
            chunk_storage_filename = metadata["chunk_storage_file"]
            zipf.extract(chunk_storage_filename)
            chunk_storage_module = importlib.import_module(
                metadata["chunk_storage"]["module"]
            )
            chunk_storage_class = getattr(
                chunk_storage_module, metadata["chunk_storage"]["class"]
            )
            chunk_storage = chunk_storage_class.load(
                chunk_storage_filename
            )  # предполагаем, что метод load в ChunkStorage работает
            os.remove(chunk_storage_filename)

            # Загрузка VectorIndex
            vector_index_filename = metadata["vector_index_file"]
            zipf.extract(vector_index_filename)
            vector_index_module = importlib.import_module(
                metadata["vector_index"]["module"]
            )
            vector_index_class = getattr(
                vector_index_module, metadata["vector_index"]["class"]
            )
            vector_index = vector_index_class.load(
                vector_index_filename
            )  # предполагаем, что метод load в FaissIndex работает
            os.remove(vector_index_filename)

            # Восстановление алгоритма чанкинга (если есть)
            chunking_algorithm = (
                None  # Это будет зависеть от вашей логики загрузки алгоритмов
            )

            # Создание объекта MirageIndex
            index = MirageIndex(
                raw_storage=raw_storage,
                chunk_storage=chunk_storage,
                chunking_algorithm=chunking_algorithm,  # Нужно добавить загрузку алгоритма чанкинга
                vector_index=vector_index,
                visualize=metadata["visualize"],
            )

            logger.info(f"Mirage index loaded from {filename}.")
            return index
