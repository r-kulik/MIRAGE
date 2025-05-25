from typing import Optional
from loguru import logger
from numpy import array, cov, expand_dims, log, ndarray, trace, zeros, float32, cumsum
from pydantic import BaseModel
from sklearn.neighbors import LocalOutlierFactor
from mirage.index import MirageIndex
from faiss import IndexHNSWFlat, IndexFlatL2, IndexFlatIP

from mirage.index.chunk_storages.WhooshChunkStorage import WhooshChunkStorage
from mirage.index.vector_index import FaissVectorIndex
from mirage.index.vector_index.VectorIndex import VectorIndex
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import skdim

logger.disable(__name__)


class IndexInvestigationResultDTO(BaseModel):
    file: Optional[str] = None
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
        self.mirage_faiss_index_object: FaissVectorIndex.FaissIndexFlatL2 = (
            mirage_index.vector_index
        )
        self.vector_index: IndexFlatL2 | IndexFlatIP = (
            self.mirage_faiss_index_object.index
        )
        self.chunk_storage: WhooshChunkStorage = mirage_index.chunk_storage
        self.vector_matrix: ndarray = None
        self.hnsw: IndexHNSWFlat = None
        self.document_labels = None
        self.LID_K = 3  # amount of neighbours for LID estimation

    def create_vector_matrix(self):
        """_summary_"""
        if self.vector_matrix is None:
            vector_dimension = self.mirage_faiss_index_object.dim
            n_vectors = self.vector_index.ntotal
            self.vector_matrix = zeros((n_vectors, vector_dimension))
            self.document_labels = zeros(n_vectors)
            document_key_label_map = {}
            counter = 0
            for i, vector_key_pair in enumerate(self.mirage_faiss_index_object):
                vector = vector_key_pair.vector
                # logger.debug(f"vector_shape = {vector.shape}, vector_matrix_shape = {self.vector_matrix.shape}")
                chunk_storage_key = vector_key_pair.chunk_storage_key
                document_key = self.chunk_storage.get_raw_index_of_document(
                    chunk_storage_key
                )
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
        return trace(cov(self.vector_matrix.T))

    def __calculate_silhouette_score_euclidian(self) -> float:
        self.create_vector_matrix()
        return silhouette_score(
            X=self.vector_matrix, labels=self.document_labels, metric="euclidean"
        )

    def __calculate_silhouette_score_cosine(self) -> float:
        self.create_vector_matrix()
        return silhouette_score(
            X=self.vector_matrix, labels=self.document_labels, metric="cosine"
        )

    def __calculate_LOFs(self) -> list[float]:
        from sklearn.neighbors import LocalOutlierFactor

        lof = LocalOutlierFactor(
            n_neighbors=self.LID_K, algorithm="kd_tree", n_jobs=-1, metric="euclidean"
        )
        lofs = lof.fit_predict(self.vector_matrix)
        logger.debug("LOFs")
        logger.debug(lofs)
        logger.debug(lofs.shape)
        return lofs

    def __calculate_LIDs(self) -> ndarray:
        self.create_vector_matrix()
        logger.debug("Start calculating")
        lpca = skdim.id.lPCA().fit_pw(
            self.vector_matrix, n_neighbors=self.LID_K, n_jobs=-1
        )
        logger.debug(lpca.dimension_pw_)
        return lpca.dimension_pw_.astype(float32)

    def calculate_EID(self) -> float:
        self.create_vector_matrix()
        pca = PCA().fit(self.vector_matrix)
        cumulative_variance = cumsum(pca.explained_variance_ratio_)
        if cumulative_variance[-1] < 0.95:
            return float(len(cumulative_variance))
        for i in range(len(cumulative_variance)):
            if cumulative_variance[i] >= 0.95:
                if i == 0:
                    return 1.0
                # Интерполяция между i-1 и i
                x0, x1 = i, i + 1
                y0, y1 = cumulative_variance[i - 1], cumulative_variance[i]
                EID = x0 + (0.95 - y0) / (y1 - y0)
                return float(EID)
        return float(len(cumulative_variance))

    def __calculate_redundancy(self, EID: int) -> float:
        return 1 - EID / self.vector_index.d

    def process(self) -> IndexInvestigationResultDTO:
        EID = self.calculate_EID()
        LOFs = self.__calculate_LOFs()

        return IndexInvestigationResultDTO(
            vector_variance=self.__calculate_variance(),
            silhouette_score_euclidian=self.__calculate_silhouette_score_euclidian(),
            silhouette_score_cosine=self.__calculate_silhouette_score_cosine(),
            EID=EID,
            redundancy=self.__calculate_redundancy(EID=EID),
            LOFs=LOFs,
            mean_LOF=LOFs.mean(),
            std_LOF=LOFs.std(),
        )
