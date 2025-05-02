
from numpy import array, cov, ndarray, trace
from pydantic import BaseModel
from mirage.index import MirageIndex
from faiss import IndexHNSWFlat, IndexFlatL2, IndexFlatIP

from mirage.index.chunk_storages.WhooshChunkStorage import WhooshChunkStorage
from mirage.index.vector_index import FaissVectorIndex
from mirage.index.vector_index.VectorIndex import VectorIndex

class IndexInvestigationResultDTO(BaseModel):
    vector_variance: float
    silhouette_score_euclidian: float
    silhouette_score_cosine: float
    EID: float
    redundancy: float
    LOFs: list[float]
    mean_LOF: float
    std_LOF: float


class IndexInvestigator:
    """
    Investigates a pure index and returns the IndexInvestigationResultDTO with statisticacl information
    about presented vector index
    """
    def __init__(self, mirage_index: MirageIndex):
        self.vector_index: IndexFlatL2 | IndexFlatIP = mirage_index.vector_index.index
        self.chunk_storage: WhooshChunkStorage = mirage_index.chunk_storage
        self.vector_matrix: ndarray = None
        self.hnsw: IndexHNSWFlat = None
        self.document_labels = None

    def create_vector_matrix(self):
        """_summary_
        """
        self.vector_matrix = self.vector_index.reconstruct_n(
            0, self.vector_index.ntotal
        )

    def create_HNSW(self):
        """
        Optimization of vector index for faster calculation of k-nearest neighbours for vectors
        """
        if self.hnsw is None:
            if self.vector_matrix is None:
                self.create_vector_matrix()
            ...
            

    def __calculate_variance(self):
        if self.vector_matrix is None:
            self.create_vector_matrix()
        return trace(
            cov(
                self.vector_matrix.T
            )
        )
    
    def __create_document_labels():
        ...

    def __calculate_silhouette_score_euclidian(self) -> float: 
        
        return 0

    def __calculate_silhouette_score_cosine(self) -> float:
        return 0
    
    def __calculate_EID(self):
        return 0
    
    def __calculate_redundancy(self, EID: int) -> float:
        return 0
    
    def __calculate_LOFs(self) -> list[float]:
        return [0, 0, 0]
    

    




    def process(self) -> IndexInvestigationResultDTO:
        EID = self.__calculate_EID()
        LOFs = self.__calculate_LOFs()
        LOFS_array = array(LOFs)
        return IndexInvestigationResultDTO(
            vector_variance=self.__calculate_variance(),
            silhouette_score_euclidian=self.__calculate_silhouette_score_euclidian(),
            silhouette_score_cosine=self.__calculate_silhouette_score_cosine(),
            EID=EID,
            redundancy=self.__calculate_redundancy(EID=EID),
            LOFs=LOFs,
            mean_LOF=LOFS_array.mean(),
            std_LOF=LOFS_array.std()
        )