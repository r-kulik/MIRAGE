from . import RawStorage, ChunkStorage, ChunkingAlgorithm, VectorIndex, QueryResult
from ..embedders import Embedder
from abc import abstractmethod, ABC
from typing import Any, final


class MirageIndex(ABC):

    def __init__(self, raw_storage, chunk_storage, chunking_algorithm, embedder, vector_index, visualize=False):
        super().__init__()
        self.raw_storage: RawStorage = raw_storage
        self.chunk_storage: ChunkStorage = chunk_storage
        self.chunking_algorithm: ChunkingAlgorithm = chunking_algorithm
        self.embedder: Embedder = embedder
        self.vector_index: VectorIndex = vector_index
        self.visualize = visualize

    @final
    def create_index(self):
        """
        Calling this function creates index of MIRAGE using the parts of it specified in the initializer
        ```py
        >>> index = MirageIndex()
        >>> index.create_index() # After this operation index.query(query_string) is allowed
        >>> index.query(query_string)
        ```
        
        """
        if self.visualize: print('Performing chunking algorithm')
        doduments_processed = self.chunking_algorithm.execute(visualize=self.visualize)
        if self.visualize: print(f"Processed {doduments_processed} documents")
        if not self.embedder.is_fitted:
            if self.visualize: print("Training an embedder...")
            self.embedder.fit(self.chunk_storage)                               # Training the embeedder if it is not trained
            self.vector_index.dim = self.embedder.get_dimensionality()          # Providing dimensionality of the embedder to the vector index
            if self.visualize: print("Embedder trained")
        if self.visualize: print("Converting chunks to vectors")
        self.embedder.convert_chunks_to_vector_index(self.chunk_storage, self.vector_index, visualize=self.visualize)
        if self.visualize: print("Creation of index has been done")          
    
    @final
    def query(self, query: str, top_k: int, return_text = False) -> list[QueryResult]:
        """
        Call this function to obtain top_k chunks of documents sematically closest to the query
        
        Args:
            query: string with query
            top_k: integer amount of chunks to be returned from the function

        Returns:
            list of QueryResult object
        """
        embedded_query = self.embedder.embed(query)
        results = self.vector_index.query(embedded_query, top_k=top_k)
        if not return_text:
            return results
        for result in results:
            result.text = self.chunk_storage[result.chunk_storage_key]
        return results
    


    @staticmethod
    def _moduleInstanceRedefining(object_passed: None | Any, 
                                  abstract_class_to_check_instance: type,
                                  default_value_to_return: Any) -> Any:
        """This function is checking the correctess of the arguments that were passed in the initializer of the `MirageIndex` subclasses' instances
        It also subctitute the default values if the were no objects passed in the initalizer or creates an instance with default arguments if there is a type \
        provided as an argument

        Parameters
        ----------
        object_passed : None | type | Any
            The value of the object that was passed as an argument for the initializer
        abstract_class_to_check_instance : type
            Abstract class the module must be inherited from
        default_value_to_return : Any
            A default value that will be returned if the object_passed is None

        Returns
        -------
        abstract_class_to_check_instance
            An object_passed that was specified if its type is a subclass of `abstract_class_to_check_instance`
            or `default_value` if `object_passed` is `None`

        Raises
        ------
        TypeError
            If type of `default_value` is not a subclass of `abstract_class_to_check_instance` 
        TypeError
            If `object_passed` type is not a subclass of `abstract_class_to_check_instance`
        """  
        print(f"object_passed = {object_passed}, its type = {type(object_passed)}\n\
              abstract = {abstract_class_to_check_instance}, default = {default_value_to_return}")
        if object_passed is None:
            if not isinstance(default_value_to_return, abstract_class_to_check_instance):
                raise TypeError(
                    f"The default value you specified to return has type {type(default_value_to_return)}, \
                        which is not a subclass of the {abstract_class_to_check_instance}"
                )
            return default_value_to_return
        if isinstance(object_passed, abstract_class_to_check_instance):
            return object_passed
        else:
            raise TypeError(
                f"The value you specified has a type {type(object_passed)} and should be a subclass of {abstract_class_to_check_instance}"
            )
        
        
        