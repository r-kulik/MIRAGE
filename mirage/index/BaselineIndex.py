from ..embedders.TextNormalizer import TextNormalizer
from . import *
from typing import Callable
from ..embedders import *
from .MirageIndex import MirageIndex
from .chunking_algorithms import ChunkingAlgorithm


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
                words_amount_in_chunk: int | None = 100,
                normalizer: bool | Callable | TextNormalizer | None = None,
                vector_index_dim: int | None = None,

                raw_storage: RawStorage | None  = None,
                chunk_storage: ChunkStorage |  None = None,
                chunking_algorithm: ChunkingAlgorithm |  None = None,
                embedder: Embedder | None = None,
                vector_index: VectorIndex |  None = None,

                visualize: bool = False
                ):
        """Creates a baseline index based on following suggestions:

            1) documents are stored in a specified folder

            2) Chunks of documents are stored in RAM

            3) chunking algorithm is a word counting algorithm

            4) embedding is done via Bag of Words algorithm 

            5) vector index is stored in RAM and it is L2 simple index

            To change suggestions redefine the modules in the arguments

        Parameters
        ----------
        data_folder : str | None, optional
            folder wgere documents are stored, by default None

        words_amount_in_chunk : int | None, optional
            amount of words presented in the chunk, size of chunk, by default None

        normalizer : bool | Callable | TextNormalizer | None, optional
            True if text normalization is needed, False and None otherwise
            Any str -> str function is allowed
            Any TextMormalizer subclass is allowed
            by default None

        vector_index_dim : int | None, optional
            dimensionality of vector index (redundant), by default None

        raw_storage : RawStorage | None, optional
            Custom RawStorage, by default None

        chunk_storage : ChunkStorage | type | None, optional
            Custom ChunkStorage, by default None

        chunking_algorithm : ChunkingAlgorithm | type | None, optional
            Custom chunking algorithm, by default None

        embedder : Embedder | None, optional
            Custom embedder, by default None

        vector_index : VectorIndex | type | None, optional
            Custom vector index, by default None

        visualize : bool, optional
            True to print stages of index creation, by default False
        """
        raw_storage = MirageIndex._moduleInstanceRedefining(
                raw_storage, 
                RawStorage, 
                FolderRawStorage(data_folder)
        )
        chunk_storage = MirageIndex._moduleInstanceRedefining(
                chunk_storage, 
                ChunkStorage, 
                RAMChunkStorage()
        )
        chunking_algorithm = MirageIndex._moduleInstanceRedefining(
            chunking_algorithm, 
            ChunkingAlgorithm, 
            WordCountingChunkingAlgorithm(raw_storage, chunk_storage, words_amount=words_amount_in_chunk)
        )
        embedder = MirageIndex._moduleInstanceRedefining(
            embedder, 
            Embedder, 
            BowEmbedder(normalizer=normalizer)
        )
        vector_index = MirageIndex._moduleInstanceRedefining(
            vector_index, 
            VectorIndex, 
            L2RAMVectorIndex(dimensionality=embedder.get_dimensionality)
        )
        
        super().__init__(
            raw_storage=raw_storage,
            chunk_storage=chunk_storage,
            chunking_algorithm=chunking_algorithm,
            embedder=embedder,
            vector_index=vector_index,
            visualize=visualize
        )