import logging
logging.basicConfig(level = logging.INFO)

from mirage.index import WordCountingChunkingAlgorithm, FolderRawStorage, SQLiteChunkStorage, RAMChunkStorage


documents = FolderRawStorage('data')
# chunks = SQLiteChunkStorage(database_name='chunks.db', table_name="chunks")
chunks = RAMChunkStorage()
algorithm = WordCountingChunkingAlgorithm(documents, chunks, words_amount=100)
algorithm.execute()
indexes = chunks.get_indexes()
print(indexes)

import random
index = random.choice(list(indexes))
print(index)
print(chunks[index])