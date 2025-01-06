from . import *
from typing import Callable
from ..embedders import *

class BaselineIndex(MirageIndex):

    """
    This class is creating an searching index from sratch basing on these assumptions:

    1) Original files are stored in a folder, its name is obligatory to provide
    2) Chunking of the files is done using Word Counting Algorithm. Each chunk has even amount of words in it.
    3) Vectorizing is done using Bag of Words algorithm (dimensionality of vector is a size of vocabulary and each item is the number of occurences of the corresponding word in a given text)
    4) Search in the vector index is done using L2 index (linear comparison of the euclidian distance between vectors)
    
    """

    def __init__(self,
                data_folder: str | None= None,
                words_amount_in_chunk: int | None = None,
                normalizer: bool | Callable | TextNormallizer | None = None,
                vector_index_dim: int | None = None,

                raw_storage: RawStorage | None  = None,
                chunk_storage: ChunkStorage | type | None = None,
                chunking_algorithm: ChunkingAlgorithm | type | None = None,
                embedder: Embedder | None = None,
                vector_index: VectorIndex | type |  None = None,

                visualize: bool = False
                ):
        """
        This function creates a basline index
        Args:
            data_folder:    path to the folder with original documents. Note that every file in the folder would be consider as a source file. To prevent this provide 
                            custom RawSrotage or custom FolderRawStorage object to the initializer. In this case data_folder argument will be ignored.

            words_amount_in_chunk: integer value representing how many words each chunk will contain

            normalizer:     Text notmalizer function, object, or flag of necessiry of normalization
                            If None or False, no normalization for text is needed
                            If True standard mirage.embedders.TextNormalizer is applied
                            Any TextNormalizer inherited object is allowed
                            Any function: str -> str that normalize text is allowed

            vector_index_dim: preferably to None, cause it nust be equal to the dimensionality of the embedder
            
            chunk_storage:      ChunkStorage object to store chunks in. By default, RAMChunkStorage will be created

            raw_storage:        RawStorage object to create index from. By default, FolderRawStorage will be created

            chunking_algorithm: ChunkingAlgorithm object or its type to create chunks from RawStorage, or type (will be created with the default parameters of superclass)

            embedder: Embedder object or its type; vectorization algorithm to convert chunks of a text into a vector

            vector_index: Vector index or its type to create from sratch

        """


        # creating a raw storage 
        if raw_storage is None and data_folder is None:
            raise ValueError(
                "You have not provided neither folder with documents nor RawStorage object to create an index"
            )
        
        # from the scratch, having only the folder name
        if raw_storage is None and data_folder is not None:
            raw_storage = FolderRawStorage(folder_path=data_folder, create_manually=False)
        
        # !-----------------------------------------------------------------------
        # creating chunk storage from the scratch if there is no chunkstorage provided
        if chunk_storage is None:
            chunk_storage = RAMChunkStorage()
        # creating a chunk_storage of a specified type
        elif issubclass(chunk_storage, ChunkStorage): 
            chunk_storage = chunk_storage()

        # !-----------------------------------------------------------------------
        if chunking_algorithm is None and words_amount_in_chunk is None:
            raise ValueError(
                "You have not provided neither amount of words in chunk neither ChunkingAlgorithm type or object to create it"
            )
        # creating chunk alforithm from the scratch if there is a words_amount_in_chunk specified
        elif chunking_algorithm is None and type(words_amount_in_chunk) == int:
            chunking_algorithm = WordCountingChunkingAlgorithm(raw_storage, chunk_storage, words_amount=words_amount_in_chunk)

        # creating a chunk algorithm object if the algorithm type was specified
        elif issubclass(chunking_algorithm, ChunkingAlgorithm):
            chunking_algorithm = chunking_algorithm(raw_storage, chunk_storage)

        # !-----------------------------------------------------------------------
        
        if embedder is None:
            embedder = BowEmbedder(normalizer=normalizer)
        elif issubclass(embedder, Embedder):
            embedder = embedder()

        # !-----------------------------------------------------------------------

        if vector_index is None:
            if embedder.is_fitted:
                vector_index = L2RAMVectorIndex(dimensionality=embedder.get_dimensionality())
            else:
                # We can not obtain dimensionality of the embedders that should be trained
                vector_index = L2RAMVectorIndex(dimensionality= -1 if vector_index_dim is None else vector_index_dim)
        
        elif issubclass(vector_index, VectorIndex):
            vector_index = vector_index(dimensionality= -1 if not embedder.is_fitted else embedder.get_dimensionality())
        
        super().__init__(
            raw_storage,
            chunk_storage,
            chunking_algorithm,
            embedder,
            vector_index,
            visualize
        )