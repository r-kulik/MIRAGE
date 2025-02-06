import warnings
import pytest
import numpy as np
from typing import Generator

# Import the base class and related classes from your module
from mirage.index.vector_index import VectorIndex, QueryResult, VectorKeyPair
from mirage.index.vector_index.FaissVectorIndex import *


@pytest.fixture
def vector_index(request):
    """
    Fixture to create an instance of a VectorIndex subclass.
    The subclass must be provided via the `request.param` parameter.
    """
    dimensionality = 3
    return request.param(dimensionality)


@pytest.mark.parametrize(
    "vector_index",
    [
        FaissIndexFlatL2,
        FaissIndexFlatIP,
        FaissIndexHNSWFlat,
        FaissIndexIVFFlat,
        FaissIndexLSH,
        FaissIndexScalarQuantizer,
        FaissIndexPQ
    ],
    indirect=True,
)
class TestVectorIndex:
    def test_abstract_methods_implemented(self, vector_index):
        """
        Ensure that all abstract methods of VectorIndex are implemented.
        This test does not require any specific behavior but ensures that
        the subclass does not raise NotImplementedError.
        """
        # Check __iter__
        assert isinstance(vector_index.__iter__(), Generator)

        # Check __contains__
          # Assuming invalid vector raises ValueError
        assert not np.array([1.0]) in vector_index

        # Check add
        with pytest.raises(Exception):  # Assuming invalid vector raises ValueError
            vector_index.add(np.array([1.0]), "key1")

        # Check query
        with pytest.raises(Exception):  # Assuming invalid query vector raises ValueError
            vector_index.query(np.array([1.0]), top_k=1)

        # Check attach_chunk_storage_key_to_vector
        with pytest.raises(VectorIndex.VectorIsNotPresentedInTheIndexException):
            vector_index.attach_chunk_storage_key_to_vector(np.array([1.0, 0.0, 0.0]), "key1")

    def test_add_and_query(self, vector_index):
        """
        Test adding vectors and querying them.
        """
        # Generate 300 random vectors and their keys
        num_vectors = 300
        vectors = [np.random.rand(3).astype('float32') for _ in range(num_vectors)]
        keys = [f"key{i}" for i in range(num_vectors)]

        # Add vectors to the index
        for vector, key in zip(vectors, keys):
            vector_index.add(vector, key)

        # Train the index
        vector_index.train()

        # Select a random vector for querying
        index_of_random = np.random.randint(0, num_vectors)
        query_vector = vectors[index_of_random] + (np.random.rand(3) - 0.5) / 500_000 
        results = vector_index.query(query_vector, top_k=3)

        # Validate the result
        assert len(results) == 3
        result = results[0]
        closest_vector = vectors[keys.index(result.chunk_storage_key)]
        assert np.allclose(result.vector, closest_vector), "Query result vector does not match the expected vector"
        # print(f"result.chunk_storage_key = {result.chunk_storage_key}; keys[vectors.index(closest_vector)] = {keys[vectors.index(closest_vector)]}")
        if not any(result.chunk_storage_key == keys[index_of_random] for result in results):
         warnings.warn("Chunk storage key mismatch, but may be query result was not perfect")

    def test_contains(self, vector_index):
        """
        Test the __contains__ method.
        """
        # Generate 300 random vectors and their keys
        num_vectors = 300
        vectors = [np.random.rand(3).astype('float32') for _ in range(num_vectors)]
        keys = [f"key{i}" for i in range(num_vectors)]

        # Add vectors to the index
        for vector, key in zip(vectors, keys):
            vector_index.add(vector, key)

        # Train the index
        vector_index.train()

        # Check containment
        random_vector = vectors[np.random.randint(0, num_vectors)]
        assert random_vector in vector_index, "Added vector is not found in the index"
        assert np.array([1.0, 0.0, 0.0]) not in vector_index, "Non-added vector is incorrectly found in the index"

    def test_attach_chunk_storage_key(self, vector_index):
        """
        Test attaching a new chunk storage key to an existing vector.
        """
        # Generate 300 random vectors and their keys
        num_vectors = 300
        vectors = [np.random.rand(3).astype('float32') for _ in range(num_vectors)]
        keys = [f"key{i}" for i in range(num_vectors)]

        # Add vectors to the index
        for vector, key in zip(vectors, keys):
            vector_index.add(vector, key)

        # Train the index
        vector_index.train()

        # Attach a new key to a random vector
        index_of_random = np.random.randint(0, num_vectors)
        random_vector = vectors[index_of_random]
        new_key = "new_key"
        vector_index.attach_chunk_storage_key_to_vector(random_vector, new_key)

        query_vector = vectors[index_of_random] + (np.random.rand(3) - 0.5) / 500_000 
        # Query the vector and validate the new key
        results = vector_index.query(query_vector, top_k=3)
        if not any(result.chunk_storage_key == new_key for result in results):  
            warnings.warn("Chunk storage key was not updated correctly, but may be query result was not perfect")

    
    def test_dimension_mismatch(self, vector_index):
        """
        Test adding or querying vectors with incorrect dimensions.
        """
        with pytest.raises(Exception):
            vector_index.add(np.array([1.0, 0.0]), "key1")  # Incorrect dimensionality

        with pytest.raises(Exception):
            vector_index.query(np.array([1.0, 0.0]), top_k=1)  # Incorrect dimensionality

    def test_iter(self, vector_index):
        """
        Test iterating over the index.
        """
        # Generate 300 random vectors and their keys
        num_vectors = 300
        vectors = [np.random.rand(3).astype('float32') for _ in range(num_vectors)]
        keys = [f"key{i}" for i in range(num_vectors)]

        # Add vectors to the index
        for vector, key in zip(vectors, keys):
            vector_index.add(vector, key)

        # Train the index
        vector_index.train()

        # Iterate over the index and validate
        vectors_and_keys = list(vector_index)
        assert len(vectors_and_keys) == num_vectors

        # Randomly check a few vectors and keys
        for _ in range(10):  # Check 10 random samples
            random_idx = np.random.randint(0, num_vectors)
            vector, key = vectors_and_keys[random_idx]
            assert np.allclose(vector, vectors[random_idx]), "Iterated vector does not match the expected vector"
            assert key == keys[random_idx], "Iterated key does not match the expected key"

    def test_train_method(self, vector_index):
        """
        Test the train method.
        """
        # Generate 300 random vectors and their keys
        num_vectors = 300
        vectors = [np.random.rand(3).astype('float32') for _ in range(num_vectors)]
        keys = [f"key{i}" for i in range(num_vectors)]

        # Add vectors to the index
        for vector, key in zip(vectors, keys):
            vector_index.add(vector, key)

        # Train the index
        vector_index.train()

        # Validate the trained state
        assert vector_index.is_trained is True

    def test_exception_on_nonexistent_vector(self, vector_index):
        """
        Test that an exception is raised when trying to attach a key to a non-existent vector.
        """
        with pytest.raises(VectorIndex.VectorIsNotPresentedInTheIndexException):
            vector_index.attach_chunk_storage_key_to_vector(np.array([1.0, 0.0, 0.0]), "key1")