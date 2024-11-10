import sqlite3

from .ChunkStorage import ChunkStorage


class SQLiteChunkStorage(ChunkStorage):

    class ProvidedTableIsInIncorrectFormat(Exception):
        def __str__(self): return "The provided table does not follow the expected format."

    def __init__(self, database_name: str, table_name: str):
        super().__init__()
        self.connection = sqlite3.connect(database_name)
        self.table_name = table_name
        self.cursor = self.connection.cursor()
        self.__create_table_if_not_exists()
        self.__validate_table_format()
        self.__read_a_table()

    def __create_table_if_not_exists(self):
        create_table_query = f"""
        CREATE TABLE IF NOT EXISTS {self.table_name} (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            text TEXT,
            raw_index_of_document TEXT
        );
        """
        self.cursor.execute(create_table_query)
        self.connection.commit()

    def __validate_table_format(self):
        # Check if the table has the correct schema
        self.cursor.execute(f"PRAGMA table_info({self.table_name});")
        columns = self.cursor.fetchall()
        expected_columns = {
            'id': 'INTEGER',
            'text': 'TEXT',
            'raw_index_of_document': 'TEXT'
        }
        for col in columns:
            col_name = col[1]
            col_type = col[2]
            if col_name in expected_columns:
                if col_type != expected_columns[col_name]:
                    raise SQLiteChunkStorage.ProvidedTableIsInIncorrectFormat()
                del expected_columns[col_name]
            else:
                raise SQLiteChunkStorage.ProvidedTableIsInIncorrectFormat()
        if expected_columns:
            raise SQLiteChunkStorage.ProvidedTableIsInIncorrectFormat()

    def __read_a_table(self):
        self.cursor.execute(f"SELECT id, text, raw_index_of_document FROM {self.table_name};")
        rows = self.cursor.fetchall()
        for row in rows:
            index = str(row[0])
            link_to_chunk = row[0]
            raw_index_of_document = row[2]
            super()._addToChunkIndex(index, link_to_chunk, raw_index_of_document)

    def add_chunk(self, text: str, raw_index_of_document: str) -> None:
        """Upload to Database a chunk"""

        insert_query = f"""
        INSERT INTO {self.table_name} (text, raw_index_of_document)
        VALUES (?, ?);
        """
        self.cursor.execute(insert_query, (text, raw_index_of_document))
        self.connection.commit()
        # Get the last inserted row id
        last_row_id = self.cursor.lastrowid
        super()._addToChunkIndex(str(last_row_id), last_row_id, raw_index_of_document)
    
    def __getitem__(self, index) -> str:
        if index in self._chunk_map:
            chunk_id = self._chunk_map[index].link_to_chunk
            self.cursor.execute(f"SELECT text FROM {self.table_name} WHERE id={chunk_id}")
            return self.cursor.fetchall()[0][0]
        else:
            raise KeyError(f"Index {index} not found in the storage.")
        
    def clear(self) -> None:
        self.cursor.execute(f"DELETE FROM {self.table_name}")
        self._chunk_map = {}