import json


with open('test_combinations.json', 'r', encoding='utf-8') as file:
    combinations = json.loads(file.read())


START_COMBINATION = 90
END_COMBINATION = len(combinations)

import datetime
from os import PathLike
import time
from typing import Dict

from loguru import logger

from mirage.embedders import HuggingFaceEmbedder
from mirage.index import MirageIndex
from mirage.index.chunk_storages import WhooshChunkStorage
from mirage.index.chunking_algorithms import WordCountingChunkingAlgorithm
from mirage.index.chunking_algorithms.NatashaSentenсeChunking import NatashaSentenceChunking
from mirage.index.raw_storages import FolderRawStorage
from mirage.index.vector_index.FaissVectorIndex import FaissIndexFlatIP, FaissIndexFlatL2

short_names = {
    'WordCountingChunkingAlgorithm': "WC_128_05_BAAI",
    'SentenceChunkingAlgorithm': 'SC'
}


def get_name(c):
    ch = 'W' if c['ChunkingAlgorithm']['method'] == 'WordCountingChunkingAlgorithm' else 'S'
    ch_par = '_'.join([str(i) for i in list(c['ChunkingAlgorithm']['params'].values())])
    e_params = c['Embedder']['params']['model'].split('/')[0]
    return 'indexes\\' + '_'.join([ch, ch_par, e_params])

get_name(combinations[12])



raw_storage = FolderRawStorage('data_txt')
def generate_index(combination: Dict, filepath_prefix: str) -> None:
    logger.info(combination)
    chunk_storage = WhooshChunkStorage(scoring_function='BM25F', normalizer=True)
    match combination['ChunkingAlgorithm']['method']:
        case 'WordCountingChunkingAlgorithm':
            chunking_algorithm = WordCountingChunkingAlgorithm(raw_storage=raw_storage, chunk_storage=chunk_storage, **combination['ChunkingAlgorithm']['params'])
        case 'SentenceChunkingAlgorithm':
            chunking_algorithm = NatashaSentenceChunking(raw_storage=raw_storage, chunk_storage=chunk_storage, **combination['ChunkingAlgorithm']['params'])
        case _:
            logger.error(combination['ChunkingAlgorithm'])
            raise ValueError('Unknown Chunking algorithm type')
    embedder = HuggingFaceEmbedder(model_name=combination['Embedder']['params']['model'])
    chunking_algorithm.execute()
    l2_index = FaissIndexFlatL2(dimensionality=embedder.get_dimensionality())
    ip_indx = FaissIndexFlatIP(dimensionality=embedder.get_dimensionality())
    embedder.convert_chunks_to_vector_index(chunk_storage=chunk_storage, vector_index=l2_index, visualize=True)
    start_copy_time = time.time()
    for vector_key_pair in l2_index:
        ip_indx.add(
            vector=vector_key_pair.vector,
            chunk_storage_key=vector_key_pair.chunk_storage_key
        )
    end_copy_time = time.time()
    logger.info(f"Copy of index time: {end_copy_time - start_copy_time}s.")
    l2_mirage = MirageIndex(
        raw_storage=raw_storage,
        chunk_storage=chunk_storage,
        chunking_algorithm=chunking_algorithm,
        vector_index=l2_index
    )
    ip_mirage = MirageIndex(
        raw_storage=raw_storage,
        chunk_storage=chunk_storage,
        chunking_algorithm=chunking_algorithm,
        vector_index=ip_indx
    )
    l2_mirage.save(filename_to_save=filepath_prefix + "_l2.mirage_index")
    ip_mirage.save(filename_to_save=filepath_prefix + "_ip.mirage_index")
    
    

for indx in range(START_COMBINATION, END_COMBINATION):
    combination = combinations[indx]
    generate_index(
        combination,
        get_name(combination)
    )