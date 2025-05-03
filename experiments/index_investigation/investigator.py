
from loguru import logger
from numpy import array, cov, expand_dims, log, ndarray, trace, zeros, float32
from pydantic import BaseModel
from sklearn.neighbors import LocalOutlierFactor
from mirage.index import MirageIndex
from faiss import IndexHNSWFlat, IndexFlatL2, IndexFlatIP

from mirage.index.chunk_storages.WhooshChunkStorage import WhooshChunkStorage
from mirage.index.vector_index import FaissVectorIndex
from mirage.index.vector_index.VectorIndex import VectorIndex
from sklearn.metrics import silhouette_score
import skdim

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
        self.mirage_faiss_index_object: FaissVectorIndex.FaissIndexFlatL2 = mirage_index.vector_index
        self.vector_index: IndexFlatL2 | IndexFlatIP = self.mirage_faiss_index_object.index
        self.chunk_storage: WhooshChunkStorage = mirage_index.chunk_storage
        self.vector_matrix: ndarray = None
        self.hnsw: IndexHNSWFlat = None
        self.document_labels = None
        self.LID_K = 3 # amount of neighbours for LID estimation 

    def create_vector_matrix(self):
        """_summary_
        """
        if self.vector_matrix is None:
            vector_dimension = self.mirage_faiss_index_object.dim
            n_vectors = self.vector_index.ntotal
            self.vector_matrix = zeros(
                (n_vectors, vector_dimension)
            )
            self.document_labels = zeros(n_vectors)
            document_key_label_map = {}
            counter = 0
            for i, vector_key_pair in enumerate(self.mirage_faiss_index_object):
                vector = vector_key_pair.vector
                # logger.debug(f"vector_shape = {vector.shape}, vector_matrix_shape = {self.vector_matrix.shape}")
                chunk_storage_key = vector_key_pair.chunk_storage_key
                document_key = self.chunk_storage.get_raw_index_of_document(chunk_storage_key)
                self.vector_matrix[i] = vector
                if document_key not in document_key_label_map:
                    document_key_label_map[document_key] = counter
                    counter += 1
                self.document_labels[i] = document_key_label_map[document_key]
        # logger.d

    # def create_HNSW(self):
    #     """
    #     Optimization of vector index for faster calculation of k-nearest neighbours for vectors
    #     """
    #     self.create_vector_matrix()
    #     self.hnsw = IndexHNSWFlat(self.vector_matrix.shape[1], 32)
    #     # self.hnsw.set_ef(100)
    #     # self.hnsw.set_ef_construction(100)
    #     # self.hnsw.set_num_neighbors(10)
    #     self.hnsw.train(self.vector_matrix)        
    #     self.hnsw.add(self.vector_matrix)
           
            
    def __calculate_variance(self):
        self.create_vector_matrix()
        return trace(
            cov(
                self.vector_matrix.T
            )
        )

    def __calculate_silhouette_score_euclidian(self) -> float:
        self.create_vector_matrix()
        return silhouette_score(
            X = self.vector_matrix,
            labels=self.document_labels,
            metric='euclidean'
        )

    def __calculate_silhouette_score_cosine(self) -> float:
        self.create_vector_matrix()
        return silhouette_score(
            X = self.vector_matrix,
            labels=self.document_labels,
            metric='cosine'
        )
    
    
    # def __calculate_LID(self, distances: ndarray) -> float:
    #     dk: float = distances[self.LID_K - 1]
    #     di: ndarray = distances[: self.LID_K - 1]
    #     return log(dk / di).sum() / (self.LID_K - 1)

    # def __calculate_LOFS(self, distances: ndarray) -> float:
    #     clf = LocalOutlierFactor(n_neighbors=self.LID)

    # def __calculate_LID_LOF(self, x: ndarray) -> float:
    #     distances, _ = self.hnsw.search(
    #         expand_dims(x, axis=0),
    #         k=self.LID_K + 1
    #     )
    #     assert distances[0][0] == 0
    #     distances = distances[0][1:]
    #     LID = self.__calculate_LID(distances)
   
    
    
    
    def __calculate_LOFs(self) -> list[float]:
        from sklearn.neighbors import LocalOutlierFactor
        lof = LocalOutlierFactor(
            n_neighbors=self.LID_K,
            algorithm='kd_tree',
            n_jobs=-1, metric='euclidean'
        )
        lofs = lof.fit_predict(self.vector_matrix)
        logger.debug('LOFs')
        logger.debug(lofs)
        logger.debug(lofs.shape)
        return lofs
    
    def __calculate_LIDs(self) -> ndarray:
        self.create_vector_matrix()
        logger.debug('Start calculating')
        lpca = skdim.id.lPCA().fit_pw(
            self.vector_matrix,
            n_neighbors=self.LID_K,
            n_jobs=-1
        )
        logger.debug(lpca.dimension_pw_)
        return lpca.dimension_pw_.astype(float32)

    def __calculate_EID(self) -> float:
        self.create_vector_matrix()
        return ((self.__calculate_LIDs() ** -1).mean() ** -1)

    
    def __calculate_redundancy(self, EID: int) -> float:
        return EID / self.vector_index.d 

    





    def process(self) -> IndexInvestigationResultDTO:
        EID = self.__calculate_EID()
        LOFs = self.__calculate_LOFs()
        
        return IndexInvestigationResultDTO(
            vector_variance=self.__calculate_variance(),
            silhouette_score_euclidian=self.__calculate_silhouette_score_euclidian(),
            silhouette_score_cosine=self.__calculate_silhouette_score_cosine(),
            EID=EID,
            redundancy=self.__calculate_redundancy(EID=EID),
            LOFs=LOFs,
            mean_LOF=LOFs.mean(),
            std_LOF=LOFs.std()
        )