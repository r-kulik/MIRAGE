from mirage import FolderRawStorage, WhooshChunkStorage, WordCountingChunkingAlgorithm
from loguru import logger
from mirage.embedders import HuggingFaceEmbedder
from mirage.index.vector_index.FaissVectorIndex import (
    FaissIndexFlatL2,
    FaissIndexFlatIP,
)
from mirage.index import MirageIndex


def main():
    logger.debug("Started script")
    raw_doc = FolderRawStorage(folder_path="data_txt")
    chunks = WhooshChunkStorage(scoring_function="BM25F", normalizer=True)
    chunking_algorithm = WordCountingChunkingAlgorithm(
        raw_storage=raw_doc, chunk_storage=chunks, words_amount=128, overlap=0.5
    )
    logger.debug("Starting chunking")
    chunking_algorithm.execute()
    emb = HuggingFaceEmbedder(model_name="BAAI/bge-m3")
    vectors = FaissIndexFlatIP(dimensionality=emb.get_dimensionality())
    logger.debug("Staring vectorizing")
    emb.convert_chunks_to_vector_index(
        chunk_storage=chunks, vector_index=vectors, visualize=True
    )
    logger.debug("Creating Index")
    index = MirageIndex(
        raw_storage=raw_doc,
        chunk_storage=chunks,
        chunking_algorithm=chunking_algorithm,
        vector_index=vectors,
    )
    logger.debug("saving index")
    index.save("main.mirage_index")
    index_loaded = MirageIndex.load("main.mirage_index")
    results = index_loaded.vector_index.query(
        query_vector=emb.embed("убийство"), top_k=10
    )
    text_results = "\n\n".join(
        index_loaded.chunk_storage.get_texts_for_search_results(results)
    )
    logger.warning(
        f"""Top 10 for query: 
{text_results}
                   """
    )


if __name__ == "__main__":
    main()
