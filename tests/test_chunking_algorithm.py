import pytest
from mirage import (
    RawStorage,
    RAMChunkStorage,
    ChunkingAlgorithm,
    WordCountingChunkingAlgorithm,
)


# Фикстура для RawStorage
@pytest.fixture
def raw_storage():
    class FakeRawStorage(RawStorage):
        def __init__(self):
            super().__init__()
            self._storage = {
                "doc1": "This is a test document with some words.",
                "doc2": "Another document for testing purposes.",
            }

        def __getitem__(self, index: str) -> str:
            return self._storage[index]

    return FakeRawStorage()


# Фикстура для RAMChunkStorage
@pytest.fixture
def ram_chunk_storage():
    return RAMChunkStorage()


# Фикстура для WordCountingChunkingAlgorithm
@pytest.fixture
def word_counting_chunking_algorithm(raw_storage, ram_chunk_storage):
    return WordCountingChunkingAlgorithm(raw_storage, ram_chunk_storage, words_amount=3)


# Общие тесты для всех реализаций ChunkingAlgorithm
def test_chunking_algorithm_initialization(word_counting_chunking_algorithm):
    assert isinstance(word_counting_chunking_algorithm, ChunkingAlgorithm)


def test_chunk_a_document(word_counting_chunking_algorithm):
    result = word_counting_chunking_algorithm.chunk_a_document("doc1")
    assert result == 1
    print(word_counting_chunking_algorithm.chunk_storage.get_indexes())
    assert len(word_counting_chunking_algorithm.chunk_storage.get_indexes()) == 3


def test_execute(word_counting_chunking_algorithm):
    result = word_counting_chunking_algorithm.execute()
    assert result == 2  # Ожидаем 2 чанка для doc1 и doc2
    assert len(word_counting_chunking_algorithm.chunk_storage.get_indexes()) == 5


def test_chunk_a_document_with_invalid_index(word_counting_chunking_algorithm):
    with pytest.raises(KeyError):
        word_counting_chunking_algorithm.chunk_a_document("nonexistent_doc")


def test_execute_with_visualization(word_counting_chunking_algorithm):
    result = word_counting_chunking_algorithm.execute(visualize=True)
    assert result == 2
    assert len(word_counting_chunking_algorithm.chunk_storage.get_indexes()) == 5
