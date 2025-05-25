import pytest
import numpy as np
from mirage import ChunkStorage, RAMChunkStorage
from mirage import L2RAMVectorIndex
from mirage import (
    BowEmbedder,
    TfIdfEmbedder,
    EmbedderIsNotTrainedException,
    HuggingFaceEmbedder,
)

# Константа для списка реализаций Embedder
EMBEDDER_IMPLEMENTATIONS = [TfIdfEmbedder, BowEmbedder]


# Фикстура для ChunkStorage
@pytest.fixture
def chunk_storage():
    class FakeChunkStorage(ChunkStorage):
        def __init__(self):
            super().__init__()
            self._storage = {
                "chunk1": "This is a test document.",
                "chunk2": "Another document for testing purposes.",
                "chunk3": "Yet another document.",
            }

        def __iter__(self):
            return iter(self._storage.items())

        def add_chunk(self, text, raw_index_of_document):
            pass

        def get_indexes(self):
            return list(self._storage.keys())

    return FakeChunkStorage()


# Фикстура для VectorIndex
@pytest.fixture
def vector_index():
    class FakeVectorIndex:
        def __init__(self):
            self.vectors = {}
            self.is_trained = True

        def add(self, vector, chunk_storage_key):
            self.vectors[chunk_storage_key] = vector

    return FakeVectorIndex()


# Параметризация для всех реализаций Embedder
@pytest.mark.parametrize("embedder_class", EMBEDDER_IMPLEMENTATIONS)
def test_embedder_initialization(embedder_class):
    embedder = embedder_class()
    assert embedder.is_fitted == False
    assert embedder.get_dimensionality() == -1


@pytest.mark.parametrize("embedder_class", EMBEDDER_IMPLEMENTATIONS)
def test_fit(embedder_class, chunk_storage):
    embedder = embedder_class()
    embedder.fit(chunk_storage)
    assert embedder.is_fitted == True
    assert embedder.get_dimensionality() > 0


@pytest.mark.parametrize("embedder_class", EMBEDDER_IMPLEMENTATIONS)
def test_embed(embedder_class, chunk_storage):
    embedder = embedder_class()
    embedder.fit(chunk_storage)
    vector = embedder.embed("This is a test document.")
    assert isinstance(vector, np.ndarray)
    assert len(vector) == embedder.get_dimensionality()


@pytest.mark.parametrize("embedder_class", EMBEDDER_IMPLEMENTATIONS)
def test_process_chunks(embedder_class, chunk_storage):
    embedder = embedder_class()
    embedder.fit(chunk_storage)
    vectors = embedder.process_chunks(chunk_storage)
    assert isinstance(vectors, dict)
    assert len(vectors.keys()) == len(chunk_storage.get_indexes())
    for key, vector in vectors.items():
        assert isinstance(vector, np.ndarray)
        assert len(vector) == embedder.get_dimensionality()


@pytest.mark.parametrize("embedder_class", EMBEDDER_IMPLEMENTATIONS)
def test_convert_chunks_to_vector_index(embedder_class, chunk_storage, vector_index):
    embedder = embedder_class()
    embedder.fit(chunk_storage)
    embedder.convert_chunks_to_vector_index(chunk_storage, vector_index)
    assert len(vector_index.vectors.keys()) == len(chunk_storage.get_indexes())
    for key, vector in vector_index.vectors.items():
        assert isinstance(vector, np.ndarray)
        assert len(vector) == embedder.get_dimensionality()


@pytest.mark.parametrize("embedder_class", EMBEDDER_IMPLEMENTATIONS)
def test_embedder_not_trained_exception(embedder_class):
    embedder = embedder_class()
    with pytest.raises(EmbedderIsNotTrainedException):
        embedder.embed("This is a test document.")
    with pytest.raises(EmbedderIsNotTrainedException):
        embedder.process_chunks(RAMChunkStorage())
    with pytest.raises(EmbedderIsNotTrainedException):
        embedder.convert_chunks_to_vector_index(
            RAMChunkStorage(), L2RAMVectorIndex(dimensionality=-1)
        )
