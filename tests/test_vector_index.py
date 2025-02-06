import pytest
import numpy as np
import os
from mirage import L2RAMVectorIndex

# Константа для списка реализаций VectorIndex
VECTOR_INDEX_IMPLEMENTATIONS = [L2RAMVectorIndex]

# Фикстура для временного файла
@pytest.fixture
def temp_file():
    return "test_vector_index.json"

# Параметризация для всех реализаций VectorIndex
@pytest.mark.parametrize("vector_index_class", VECTOR_INDEX_IMPLEMENTATIONS)
def test_vector_index_initialization(vector_index_class):
    index = vector_index_class(dimensionality=3)
    assert index.dim == 3

@pytest.mark.parametrize("vector_index_class", VECTOR_INDEX_IMPLEMENTATIONS)
def test_add(vector_index_class):
    index = vector_index_class(dimensionality=3)
    vector = np.array([1, 2, 3])
    index.add(vector, "chunk1")
    assert vector in index

@pytest.mark.parametrize("vector_index_class", VECTOR_INDEX_IMPLEMENTATIONS)
def test_contains(vector_index_class):
    index = vector_index_class(dimensionality=3)
    vector = np.array([1, 2, 3])
    index.add(vector, "chunk1")
    assert vector in index
    assert np.array([4, 5, 6]) not in index

@pytest.mark.parametrize("vector_index_class", VECTOR_INDEX_IMPLEMENTATIONS)
def test_iter(vector_index_class):
    index = vector_index_class(dimensionality=3)
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    index.add(vector1, "chunk1")
    index.add(vector2, "chunk2")
    vectors = [vkp.vector for vkp in index]
    
    assert any(np.array_equal(vector2, arr) for arr in vectors)
    assert  any(np.array_equal(vector2, arr) for arr in vectors)

@pytest.mark.parametrize("vector_index_class", VECTOR_INDEX_IMPLEMENTATIONS)
def test_query(vector_index_class):
    index = vector_index_class(dimensionality=3)
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    index.add(vector1, "chunk1")
    index.add(vector2, "chunk2")
    results = index.query(np.array([1, 2, 3]), top_k=2)
    assert len(results) == 2
    assert results[0].vector.tolist() == [1, 2, 3]
    assert results[0].chunk_storage_key == "chunk1"

@pytest.mark.parametrize("vector_index_class", VECTOR_INDEX_IMPLEMENTATIONS)
def test_attach_chunk_storage_key_to_vector(vector_index_class):
    index = vector_index_class(dimensionality=3)
    vector = np.array([1, 2, 3])
    index.add(vector, "chunk1")
    index.attach_chunk_storage_key_to_vector(vector, "chunk2")
    results = index.query(vector, top_k=1)
    assert results[0].chunk_storage_key == "chunk2"


@pytest.mark.parametrize("vector_index_class", VECTOR_INDEX_IMPLEMENTATIONS)
def test_save_and_load(vector_index_class, temp_file):
    index = vector_index_class(dimensionality=3)
    vector1 = np.array([1, 2, 3])
    vector2 = np.array([4, 5, 6])
    index.add(vector1, "chunk1")
    index.add(vector2, "chunk2")
    index.save(temp_file)

    new_index = vector_index_class(dimensionality=3)
    print(temp_file)
    new_index.load(temp_file)
    assert vector1 in new_index
    assert vector2 in new_index
    results = new_index.query(vector1, top_k=1)
    assert results[0].chunk_storage_key == "chunk1"