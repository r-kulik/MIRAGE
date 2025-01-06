import pytest
from mirage import ChunkStorage, RAMChunkStorage, SQLiteChunkStorage
import os

fixtures = ["ram_chunk_storage", "sqlite_chunk_storage"] 


# Фикстура для RAMChunkStorage
@pytest.fixture
def ram_chunk_storage():
    return RAMChunkStorage()

# Фикстура для SQLiteChunkStorage
@pytest.fixture
def sqlite_chunk_storage(tmpdir):
    db_path = os.path.join(tmpdir, "test.db")
    return SQLiteChunkStorage(db_path, "chunks")

# Общие тесты для всех реализаций ChunkStorage
@pytest.mark.parametrize("storage_fixture", fixtures)
def test_chunk_storage_initialization(request, storage_fixture):
    storage = request.getfixturevalue(storage_fixture)
    assert isinstance(storage, ChunkStorage)
    assert storage.get_indexes() == []

@pytest.mark.parametrize("storage_fixture", fixtures)
def test_add_chunk(request, storage_fixture):
    storage = request.getfixturevalue(storage_fixture)
    index = storage.add_chunk("test chunk", "doc1")
    assert index in storage.get_indexes()
    assert storage.get_raw_index_of_document(index) == "doc1"
    assert storage[index] == "test chunk"


@pytest.mark.parametrize("storage_fixture", fixtures)
def test_clear_storage(request, storage_fixture):
    storage = request.getfixturevalue(storage_fixture)
    storage.add_chunk("test chunk", "doc1")
    storage.clear()
    assert storage.get_indexes() == []

@pytest.mark.parametrize("storage_fixture", fixtures)
def test_iter_storage(request, storage_fixture):
    storage = request.getfixturevalue(storage_fixture)
    storage.add_chunk("test chunk 1", "doc1")
    storage.add_chunk("test chunk 2", "doc2")
    chunks = list(storage)
    print("chunks: ",  chunks)
    assert len(chunks) == 2
    assert any(chunk[1] == "test chunk 1" for chunk in chunks)
    assert any(chunk[1] == "test chunk 2" for chunk in chunks)

@pytest.mark.parametrize("storage_fixture", fixtures)
def test_get_nonexistent_chunk(request, storage_fixture):
    storage = request.getfixturevalue(storage_fixture)
    with pytest.raises(KeyError):
        storage["nonexistent_index"]