import faiss
import numpy as np
from typing import Dict, List, Literal, Tuple, Generator

from mirage.index.vector_index import VectorIndex
from mirage.index.vector_index.VectorIndex import QueryResult, VectorKeyPair


class FaissIndexFlatL2(VectorIndex):
    """
    Implementation of VectorIndex using FAISS's IndexFlatL2.
    """

    def __init__(self, dimensionality: int):
        super().__init__(dimensionality)
        self.index = faiss.IndexFlatL2(dimensionality)
        self.vector_to_key_map = {}
        self._buffer = []  # Buffer for vectors and keys
        self.is_trained = True  # FlatIP does not require training

    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        for i in range(self.index.ntotal):
            vector = self.index.reconstruct(i)
            chunk_storage_key = self.vector_to_key_map[tuple(vector)]
            yield VectorKeyPair(vector, chunk_storage_key)

    def __contains__(self, vector: np.ndarray) -> bool:
        return tuple(vector) in self.vector_to_key_map

    def add(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimensionality {vector.shape[0]} does not match index dimensionality {self.dim}")
        if tuple(vector) in self.vector_to_key_map:
            raise ValueError(f"Vector {vector} is already present in the index")

        if self.is_trained:
            self.index.add(np.expand_dims(vector, axis=0))
            self.vector_to_key_map[tuple(vector)] = chunk_storage_key
        else:
            # Buffer the vector and key if the index is not trained
            self._buffer.append((vector, chunk_storage_key))

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[QueryResult]:
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"Query vector dimensionality {query_vector.shape[0]} does not match index dimensionality {self.dim}")
        
        distances, indices = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result found
                break
            vector = self.index.reconstruct(int(idx))
            chunk_storage_key = self.vector_to_key_map[tuple(vector)]
            results.append(QueryResult(distance=dist, vector=vector, chunk_storage_key=chunk_storage_key))
        return results

    def attach_chunk_storage_key_to_vector(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        vector_tuple = tuple(vector)
        if vector_tuple not in self.vector_to_key_map:
            raise self.VectorIsNotPresentedInTheIndexException(vector)
        self.vector_to_key_map[vector_tuple] = chunk_storage_key

    def train(self) -> None:
        if not self.is_trained:
            # FlatL2 does not require explicit training, but we need to add buffered vectors
            for vector, chunk_storage_key in self._buffer:
                self.index.add(np.expand_dims(vector, axis=0))
                self.vector_to_key_map[tuple(vector)] = chunk_storage_key
            self._buffer.clear()
            self.is_trained = True


class FaissIndexFlatIP(VectorIndex):
    """
    Implementation of VectorIndex using FAISS's IndexFlatIP .
    """

    def __init__(self, dimensionality: int):
        super().__init__(dimensionality)
        self.index = faiss.IndexFlatIP(dimensionality)
        self.vector_to_key_map = {}
        self._buffer = []  # Buffer for vectors and keys
        self.is_trained = True  # FlatIP does not require training

    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        for i in range(self.index.ntotal):
            vector = self.index.reconstruct(i)
            chunk_storage_key = self.vector_to_key_map[tuple(vector)]
            yield VectorKeyPair(vector, chunk_storage_key)

    def __contains__(self, vector: np.ndarray) -> bool:
        return tuple(vector) in self.vector_to_key_map

    def add(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimensionality {vector.shape[0]} does not match index dimensionality {self.dim}")
        if tuple(vector) in self.vector_to_key_map:
            raise ValueError(f"Vector {vector} is already present in the index")

        if self.is_trained:
            self.index.add(np.expand_dims(vector, axis=0))
            self.vector_to_key_map[tuple(vector)] = chunk_storage_key
        else:
            # Buffer the vector and key if the index is not trained
            self._buffer.append((vector, chunk_storage_key))

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[QueryResult]:
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"Query vector dimensionality {query_vector.shape[0]} does not match index dimensionality {self.dim}")
        
        similarities, indices = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # No result found
                break
            vector = self.index.reconstruct(int(idx))
            chunk_storage_key = self.vector_to_key_map[tuple(vector)]
            results.append(QueryResult(distance=sim, vector=vector, chunk_storage_key=chunk_storage_key))
        return results

    def attach_chunk_storage_key_to_vector(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        vector_tuple = tuple(vector)
        if vector_tuple not in self.vector_to_key_map:
            raise self.VectorIsNotPresentedInTheIndexException(vector)
        self.vector_to_key_map[vector_tuple] = chunk_storage_key

    def train(self) -> None:
        if not self.is_trained:
            # FlatIP does not require explicit training, but we need to add buffered vectors
            for vector, chunk_storage_key in self._buffer:
                self.index.add(np.expand_dims(vector, axis=0))
                self.vector_to_key_map[tuple(vector)] = chunk_storage_key
            self._buffer.clear()
            self.is_trained = True


class FaissIndexHNSWFlat(VectorIndex):
    """
    Implementation of VectorIndex using FAISS's IndexHNSWFlat.
    """

    def __init__(self, dimensionality: int, M: int = 16, efConstruction: int = 40):
        """
        Args:
            dimensionality: Dimensionality of the vector space.
            M: The number of bi-directional links created for each point in the graph.
            efConstruction: Size of the dynamic list used during construction.
        """
        super().__init__(dimensionality)
        self.index = faiss.IndexHNSWFlat(dimensionality, M)
        self.index.hnsw.efConstruction = efConstruction
        self.vector_to_key_map = {}
        self._buffer = []  # Buffer for vectors and keys
        self.is_trained = False

    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        for i in range(self.index.ntotal):
            vector = self.index.reconstruct(i)
            chunk_storage_key = self.vector_to_key_map[tuple(vector)]
            yield VectorKeyPair(vector, chunk_storage_key)

    def __contains__(self, vector: np.ndarray) -> bool:
        return tuple(vector) in self.vector_to_key_map

    def add(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimensionality {vector.shape[0]} does not match index dimensionality {self.dim}")
        if tuple(vector) in self.vector_to_key_map:
            raise ValueError(f"Vector {vector} is already present in the index")

        if self.is_trained:
            self.index.add(np.expand_dims(vector, axis=0))
            self.vector_to_key_map[tuple(vector)] = chunk_storage_key
        else:
            # Buffer the vector and key if the index is not trained
            self._buffer.append((vector, chunk_storage_key))

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[QueryResult]:
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"Query vector dimensionality {query_vector.shape[0]} does not match index dimensionality {self.dim}")

        distances, indices = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result found
                break
            vector = self.index.reconstruct(int(idx))
            chunk_storage_key = self.vector_to_key_map[tuple(vector)]
            results.append(QueryResult(distance=dist, vector=vector, chunk_storage_key=chunk_storage_key))
        return results

    def attach_chunk_storage_key_to_vector(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        vector_tuple = tuple(vector)
        if vector_tuple not in self.vector_to_key_map:
            raise self.VectorIsNotPresentedInTheIndexException(vector)
        self.vector_to_key_map[vector_tuple] = chunk_storage_key

    def train(self) -> None:
        if not self.is_trained:
            for vector, chunk_storage_key in self._buffer:
                self.index.add(np.expand_dims(vector, axis=0))
                self.vector_to_key_map[tuple(vector)] = chunk_storage_key
            self._buffer.clear()
            self.is_trained = True


class FaissIndexIVFFlat(VectorIndex):
    """
    Implementation of VectorIndex using FAISS's IndexIVFFlat.
    """

    def __init__(self, dimensionality: int, nlist: int = 1, metric: int = faiss.METRIC_L2):
        super().__init__(dimensionality)
        self.quantizer = faiss.IndexFlatL2(dimensionality)
        self.index = faiss.IndexIVFFlat(self.quantizer, dimensionality, nlist, metric)
        self.vector_to_key_map = {}  # Maps vectors (as tuple) to chunk_storage_keys
        self._buffer = []  # Buffer for vectors and keys
        self.is_trained = False


    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        if not self.is_trained:
            raise RuntimeError("Index is not trained. Call `train()` before iterating.")

        # Use the internal mapping to iterate over vectors and keys
        for vector_tuple, chunk_storage_key in self.vector_to_key_map.items():
            yield VectorKeyPair(np.array(vector_tuple), chunk_storage_key)

    def __contains__(self, vector: np.ndarray) -> bool:
        return tuple(vector) in self.vector_to_key_map

    def add(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimensionality {vector.shape[0]} does not match index dimensionality {self.dim}")
        if tuple(vector) in self.vector_to_key_map:
            raise ValueError(f"Vector {vector} is already present in the index")

        if self.is_trained:
            self.index.add(np.expand_dims(vector, axis=0))
            self.vector_to_key_map[tuple(vector)] = chunk_storage_key
        else:
            self._buffer.append((vector, chunk_storage_key))    

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[QueryResult]:
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"Query vector dimensionality {query_vector.shape[0]} does not match index dimensionality {self.dim}")

        distances, indices = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result found
                break
            vector = self.index.reconstruct(int(idx))
            chunk_storage_key = self.vector_to_key_map[tuple(vector)]
            results.append(QueryResult(distance=dist, vector=vector, chunk_storage_key=chunk_storage_key))
        return results

    def attach_chunk_storage_key_to_vector(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        vector_tuple = tuple(vector)
        if vector_tuple not in self.vector_to_key_map:
            raise self.VectorIsNotPresentedInTheIndexException(vector)
        self.vector_to_key_map[vector_tuple] = chunk_storage_key

    def train(self) -> None:
        if not self.is_trained:
            if not self._buffer:
                raise RuntimeError("Cannot train the index because the buffer is empty. Add vectors first.")

            # Extract vectors from the buffer for training
            training_data = np.array([vector for vector, _ in self._buffer], dtype=np.float32)

            # Train the index
            self.index.train(training_data)

            # Add all vectors from the buffer to the index
            for vector, chunk_storage_key in self._buffer:
                self.index.add(np.expand_dims(vector, axis=0))
                self.vector_to_key_map[tuple(vector)] = chunk_storage_key

            faiss.downcast_index(self.index).make_direct_map()  # ГОСПОДИ ЧТО ЭТО
            # Clear the buffer
            self._buffer.clear()
            self.is_trained = True



class FaissIndexIVFFlat(VectorIndex):
    """
    Implementation of VectorIndex using FAISS's IndexIVFFlat.
    """

    def __init__(self, dimensionality: int, nlist: int = 1, metric: int = faiss.METRIC_L2):
        """
        Args:
            dimensionality: Dimensionality of the vector space.
            nlist: Number of clusters (centroids) in the index.
            metric: Distance metric to use (faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT).
        """
        super().__init__(dimensionality)
        self.quantizer = faiss.IndexFlatL2(dimensionality)
        self.index = faiss.IndexIVFFlat(self.quantizer, dimensionality, nlist, metric)

        # Enable maintain_direct_map directly on the index
        if isinstance(self.index, faiss.IndexIVF):  # Ensure it's an IndexIVF instance
            self.index.maintain_direct_map = True

        self.vector_to_key_map = {}  # Maps vectors (as tuple) to chunk_storage_keys
        self._buffer = []  # Buffer for vectors and keys
        self.is_trained = False

    def add(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        """
        Add a vector to the index.
        If the index is not trained, buffer the vector until training is complete.
        """
        if vector.shape[0] != self.dim:
            raise ValueError(f"Vector dimensionality {vector.shape[0]} does not match index dimensionality {self.dim}")
        if tuple(vector) in self.vector_to_key_map:
            raise ValueError(f"Vector {vector} is already present in the index")

        if self.is_trained:
            self.index.add(np.expand_dims(vector, axis=0))
            self.vector_to_key_map[tuple(vector)] = chunk_storage_key
        else:
            self._buffer.append((vector, chunk_storage_key))

    def train(self) -> None:
        """
        Train the index using buffered vectors.
        After training, add all buffered vectors to the index.
        """
        if not self.is_trained:
            if not self._buffer:
                raise RuntimeError("Cannot train the index because the buffer is empty. Add vectors first.")

            # Extract vectors from the buffer for training
            training_data = np.array([vector for vector, _ in self._buffer], dtype=np.float32)

            # Train the index
            self.index.train(training_data)

            # Add all vectors from the buffer to the index
            for vector, chunk_storage_key in self._buffer:
                self.index.add(np.expand_dims(vector, axis=0))
                self.vector_to_key_map[tuple(vector)] = chunk_storage_key
            faiss.downcast_index(self.index).make_direct_map()  # ГОСПОДИ ЧТО ЭТО
            # Clear the buffer
            self._buffer.clear()
            self.is_trained = True

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[QueryResult]:
        """
        Query the index for the top-k nearest neighbors.
        """
        if query_vector.shape[0] != self.dim:
            raise ValueError(f"Query vector dimensionality {query_vector.shape[0]} does not match index dimensionality {self.dim}")

        distances, indices = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
        results = []

        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result found
                break

            # Use reconstruct to retrieve the vector for the given index
            try:
                vector = self.index.reconstruct(int(idx))
                vector_tuple = tuple(vector)
                chunk_storage_key = self.vector_to_key_map.get(vector_tuple, None)

                if chunk_storage_key is None:
                    raise RuntimeError(f"Could not find chunk storage key for vector: {vector}")

                results.append(QueryResult(distance=dist, vector=vector, chunk_storage_key=chunk_storage_key))
            except Exception as e:
                raise RuntimeError(f"Error while reconstructing vector for index {idx}: {e}")

        return results

    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        """
        Iterate over the index, yielding vector-key pairs.
        """
        if not self.is_trained:
            raise RuntimeError("Index is not trained. Call `train()` before iterating.")

        # Use the internal mapping to iterate over vectors and keys
        for vector_tuple, chunk_storage_key in self.vector_to_key_map.items():
            yield VectorKeyPair(np.array(vector_tuple), chunk_storage_key)

    def __contains__(self, vector: np.ndarray) -> bool:
        """
        Check if a vector is present in the index.
        """
        return tuple(vector) in self.vector_to_key_map

    def attach_chunk_storage_key_to_vector(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        """
        Attach a new chunk storage key to an existing vector.
        """
        vector_tuple = tuple(vector)
        if vector_tuple not in self.vector_to_key_map:
            raise self.VectorIsNotPresentedInTheIndexException(vector)
        self.vector_to_key_map[vector_tuple] = chunk_storage_key


class FaissIndexLSH(VectorIndex):
    """
    Implementation of VectorIndex using FAISS's IndexLSH with local vector storage.
    """

    def __init__(self, dimensionality: int, nbits: int = 32):
        """
        Args:
            dimensionality: Dimensionality of the vector space.
            nbits: Number of bits for hash functions.
        """
        super().__init__(dimensionality)
        self.index = faiss.IndexLSH(dimensionality, nbits)
        
        # Local storage for vectors and keys
        self.stored_vectors = []  # Stores vectors as np.float32
        self.storage_keys = []    # Corresponding chunk storage keys
        self.vector_to_index = {}  # Maps tuple(vector) to storage index
        self.vector_to_key = {}    # Maps tuple(vector) to chunk key

        self.is_trained = True

    def add(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        # Normalize vector shape and type
        vector = vector.reshape(-1).astype(np.float32)
        if len(vector) != self.dim:
            raise ValueError(f"Vector dim {len(vector)} ≠ index dim {self.dim}")
            
        vector_tuple = tuple(vector)
        if vector_tuple in self.vector_to_key:
            raise ValueError(f"Vector {vector} already exists in index")

        # Add to FAISS index (requires 2D array)
        self.index.add(np.expand_dims(vector, axis=0))
        
        # Update local storage
        index = len(self.stored_vectors)
        self.stored_vectors.append(vector)
        self.storage_keys.append(chunk_storage_key)
        self.vector_to_index[vector_tuple] = index
        self.vector_to_key[vector_tuple] = chunk_storage_key

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[QueryResult]:
        # Prepare query vector
        query_vector = query_vector.reshape(-1).astype(np.float32)
        if len(query_vector) != self.dim:
            raise ValueError(f"Query vector dim {len(query_vector)} ≠ index dim {self.dim}")

        # Perform search
        distances, indices = self.index.search(
            np.expand_dims(query_vector, axis=0), 
            top_k
        )
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:  # No result
                continue
                
            try:
                # Get data from local storage
                vector = self.stored_vectors[idx]
                key = self.storage_keys[idx]
                
                # Validate consistency
                if self.vector_to_key[tuple(vector)] != key:
                    raise RuntimeError("Data inconsistency detected")
                    
                results.append(QueryResult(
                    distance=float(dist),
                    vector=vector,
                    chunk_storage_key=key
                ))
            except (IndexError, KeyError) as e:
                raise RuntimeError(f"Invalid index {idx}: {str(e)}")

        return results

    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        """Iterate over all stored vector-key pairs"""
        for vec, key in zip(self.stored_vectors, self.storage_keys):
            yield VectorKeyPair(vec.copy(), key)

    def __contains__(self, vector: np.ndarray) -> bool:
        vector = vector.reshape(-1).astype(np.float32)
        return tuple(vector) in self.vector_to_key

    def attach_chunk_storage_key_to_vector(self, 
                                         vector: np.ndarray, 
                                         chunk_storage_key: str) -> None:
        vector = vector.reshape(-1).astype(np.float32)
        vector_tuple = tuple(vector)
        
        if vector_tuple not in self.vector_to_index:
            raise self.VectorIsNotPresentedInTheIndexException(vector)
            
        index = self.vector_to_index[vector_tuple]
        self.storage_keys[index] = chunk_storage_key
        self.vector_to_key[vector_tuple] = chunk_storage_key

    def train(self):
        self.is_trained = True


import numpy as np
import faiss

class FaissIndexScalarQuantizer(VectorIndex):
    """
    Implementation of VectorIndex using FAISS's IndexScalarQuantizer with training support.
    """

    def __init__(self, dimensionality: int, quantizer_type: str | faiss.Quantizer = "QT_8bit", metric: int = faiss.METRIC_L2):
        super().__init__(dimensionality)
        
        # Convert quantizer_type to FAISS enum
        self.qtype = self._parse_quantizer_type(quantizer_type)
        
        # Initialize index
        self.index = faiss.IndexScalarQuantizer(dimensionality, self.qtype, metric)
        
        # Training state management
        self._buffer = []
        self.is_trained = False
        
        # Storage for vectors and keys
        self.stored_vectors = []  # type: List[np.ndarray]
        self.storage_keys = []    # type: List[str]
        self.vector_to_index = {} # type: Dict[tuple, int]  # Добавлен недостающий атрибут
        self.vector_to_key = {}   # type: Dict[tuple, str]

    def _parse_quantizer_type(self, quantizer_type: str) -> int:
        """Convert string quantizer type to FAISS enum value"""
        type_map = {
            "QT_8bit": faiss.ScalarQuantizer.QT_8bit,
            "QT_4bit": faiss.ScalarQuantizer.QT_4bit,
            "QT_6bit": faiss.ScalarQuantizer.QT_6bit,
            "QT_fp16": faiss.ScalarQuantizer.QT_fp16
        }
        return type_map.get(quantizer_type, faiss.ScalarQuantizer.QT_8bit)

    def add(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        vector = vector.reshape(-1).astype(np.float32)
        if len(vector) != self.dim:
            raise ValueError(f"Vector dim {len(vector)} ≠ index dim {self.dim}")
            
        vector_tuple = tuple(vector)
        if vector_tuple in self.vector_to_key:
            raise ValueError(f"Vector {vector} already exists in index")

        if self.is_trained:
            self.index.add(np.expand_dims(vector, axis=0))
            index = len(self.stored_vectors)
            self.stored_vectors.append(vector)
            self.storage_keys.append(chunk_storage_key)
            self.vector_to_index[vector_tuple] = index  # Исправлено: использование vector_to_index
            self.vector_to_key[vector_tuple] = chunk_storage_key
        else:
            self._buffer.append((vector, chunk_storage_key))



    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[QueryResult]:
        if not self.is_trained:
            raise RuntimeError("Index must be trained before querying")

        query_vector = query_vector.reshape(-1).astype(np.float32)
        if len(query_vector) != self.dim:
            raise ValueError(f"Query vector dim {len(query_vector)} ≠ index dim {self.dim}")

        distances, indices = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
                
            try:
                vector = self.stored_vectors[idx]
                key = self.storage_keys[idx]
                results.append(QueryResult(
                    distance=float(dist),
                    vector=vector,
                    chunk_storage_key=key
                ))
            except IndexError:
                raise RuntimeError(f"Invalid index {idx} in search results")
        
        return results

    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        """Iterate over all stored vector-key pairs"""
        for vec, key in zip(self.stored_vectors, self.storage_keys):
            yield VectorKeyPair(vec.copy(), key)

    def __contains__(self, vector: np.ndarray) -> bool:
        vector = vector.reshape(-1).astype(np.float32)
        return tuple(vector) in self.vector_to_key

    def attach_chunk_storage_key_to_vector(self, 
                                         vector: np.ndarray, 
                                         chunk_storage_key: str) -> None:
        vector = vector.reshape(-1).astype(np.float32)
        vector_tuple = tuple(vector)
        
        if vector_tuple not in self.vector_to_index:  # Теперь используем vector_to_index
            raise self.VectorIsNotPresentedInTheIndexException(vector)
            
        index = self.vector_to_index[vector_tuple]
        self.storage_keys[index] = chunk_storage_key
        self.vector_to_key[vector_tuple] = chunk_storage_key

    def train(self) -> None:
        """Train the index using buffered vectors"""
        if not self._buffer:
            raise RuntimeError("Need at least one vector to train the index")

        # Extract training data
        training_data = np.array([v for v, _ in self._buffer], dtype=np.float32)
        
        # ScalarQuantizer requires training
        self.index.train(training_data)
        
        # Add buffered vectors
        for vec, key in self._buffer:
            self.index.add(np.expand_dims(vec, axis=0))
            index = len(self.stored_vectors)
            self.stored_vectors.append(vec)
            self.storage_keys.append(key)
            self.vector_to_index[tuple(vec)] = index  # Исправлено: сохранение индекса
            self.vector_to_key[tuple(vec)] = key
        
        self._buffer.clear()
        self.is_trained = True


class FaissIndexPQ(VectorIndex):
    """
    Implementation of VectorIndex using FAISS's IndexPQ (Product Quantization) 
    with local vector storage and training support.
    """

    def __init__(self, dimensionality: int, M: int = 1, nbits: int = 8, 
                 metric: int = faiss.METRIC_L2):
        """
        Args:
            dimensionality: Dimensionality of the vector space
            M: Number of sub-quantizers (must divide dimensionality)
            nbits: Number of bits per sub-quantizer index
            metric: Distance metric (faiss.METRIC_L2 or faiss.METRIC_INNER_PRODUCT)
        """
        super().__init__(dimensionality)
        
        # Validate parameters
        if dimensionality % M != 0:
            raise ValueError(f"Dimensionality {dimensionality} must be divisible by M={M}")
        
        # Initialize PQ index
        self.index = faiss.IndexPQ(dimensionality, M, nbits, metric)
        
        # Training state management
        self._buffer = []
        self.is_trained = False
        
        # Local storage for original vectors and keys
        self.stored_vectors = []  # type: List[np.ndarray]
        self.storage_keys = []    # type: List[str]
        self.vector_to_index = {} # type: Dict[tuple, int]
        self.vector_to_key = {}   # type: Dict[tuple, str]

    def add(self, vector: np.ndarray, chunk_storage_key: str) -> None:
        vector = vector.reshape(-1).astype(np.float32)
        if len(vector) != self.dim:
            raise ValueError(f"Vector dim {len(vector)} ≠ index dim {self.dim}")
            
        vector_tuple = tuple(vector)
        if vector_tuple in self.vector_to_key:
            raise ValueError(f"Vector {vector} already exists in index")

        if self.is_trained:
            # Direct addition to trained index
            self.index.add(np.expand_dims(vector, axis=0))
            index = len(self.stored_vectors)
            self.stored_vectors.append(vector)
            self.storage_keys.append(chunk_storage_key)
            self.vector_to_index[vector_tuple] = index
            self.vector_to_key[vector_tuple] = chunk_storage_key
        else:
            # Buffer vectors until training
            self._buffer.append((vector, chunk_storage_key))

    def train(self) -> None:
        """Train the PQ index using buffered vectors"""
        if not self._buffer:
            raise RuntimeError("Need training data. Add vectors before calling train()")
        
        # Convert buffer to training data
        training_data = np.array([v for v, _ in self._buffer], dtype=np.float32)
        
        # PQ requires specific training logic
        if training_data.shape[0] < 256:  # Minimum for PQ training
            raise RuntimeError(f"Need at least 256 vectors for PQ training, got {training_data.shape[0]}")
        
        # Train the index
        self.index.train(training_data)
        
        # Add buffered vectors after training
        for vec, key in self._buffer:
            vec = vec.astype(np.float32)
            self.index.add(np.expand_dims(vec, axis=0))
            index = len(self.stored_vectors)
            self.stored_vectors.append(vec)
            self.storage_keys.append(key)
            self.vector_to_index[tuple(vec)] = index
            self.vector_to_key[tuple(vec)] = key
        
        self._buffer.clear()
        self.is_trained = True

    def query(self, query_vector: np.ndarray, top_k: int = 1) -> List[QueryResult]:
        if not self.is_trained:
            raise RuntimeError("Index must be trained before querying")
            
        query_vector = query_vector.reshape(-1).astype(np.float32)
        if len(query_vector) != self.dim:
            raise ValueError(f"Query vector dim {len(query_vector)} ≠ index dim {self.dim}")

        distances, indices = self.index.search(np.expand_dims(query_vector, axis=0), top_k)
        results = []
        
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
                
            try:
                vector = self.stored_vectors[idx]
                key = self.storage_keys[idx]
                results.append(QueryResult(
                    distance=float(dist),
                    vector=vector,
                    chunk_storage_key=key
                ))
            except IndexError:
                raise RuntimeError(f"Invalid index {idx} in search results")
        
        return results

    def __iter__(self) -> Generator[VectorKeyPair, None, None]:
        """Iterate over all stored vector-key pairs"""
        for vec, key in zip(self.stored_vectors, self.storage_keys):
            yield VectorKeyPair(vec.copy(), key)

    def __contains__(self, vector: np.ndarray) -> bool:
        vector = vector.reshape(-1).astype(np.float32)
        return tuple(vector) in self.vector_to_key

    def attach_chunk_storage_key_to_vector(self, 
                                         vector: np.ndarray, 
                                         chunk_storage_key: str) -> None:
        vector = vector.reshape(-1).astype(np.float32)
        vector_tuple = tuple(vector)
        
        if vector_tuple not in self.vector_to_index:
            raise self.VectorIsNotPresentedInTheIndexException(vector)
            
        index = self.vector_to_index[vector_tuple]
        self.storage_keys[index] = chunk_storage_key
        self.vector_to_key[vector_tuple] = chunk_storage_key

