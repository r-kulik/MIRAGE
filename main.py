import logging
logging.basicConfig(level = logging.INFO)

from mirage.index import WordCountingChunkingAlgorithm, FolderRawStorage, SQLiteChunkStorage


documents = FolderRawStorage('data')
chunks = SQLiteChunkStorage(database_name='chunks.db', table_name="chunks")
algorithm = WordCountingChunkingAlgorithm(documents, chunks, words_amount=100)
# algorithm.execute()
print(chunks["234"])