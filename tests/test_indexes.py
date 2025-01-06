import pytest

from mirage import MirageIndex, BaselineIndex
from mirage import FolderRawStorage
from mirage import RAMChunkStorage
from mirage import WordCountingChunkingAlgorithm
from mirage import BowEmbedder, TextNormalizer
from mirage import L2RAMVectorIndex

# Константа для списка реализаций MirageIndex
MIRAGE_INDEX_IMPLEMENTATIONS = [BaselineIndex]

# Фикстура для временной папки с документами
@pytest.fixture
def temp_data_folder(tmpdir):
    # Создаём временные файлы
    file1 = tmpdir.join("doc1.txt")
    file1.write("This is a test document.")
    file2 = tmpdir.join("doc2.txt")
    file2.write("Another document for testing purposes.")
    return str(tmpdir)

# Фикстура для класса BaselineIndex
@pytest.fixture
def baseline_index_class():
    return BaselineIndex

# Параметризация для всех реализаций MirageIndex
@pytest.mark.parametrize("index_class", MIRAGE_INDEX_IMPLEMENTATIONS)
def test_mirage_index_initialization(index_class, temp_data_folder):
    index = index_class(data_folder=temp_data_folder, words_amount_in_chunk=3, visualize=True)
    assert isinstance(index, MirageIndex)
    assert isinstance(index.raw_storage, FolderRawStorage)
    assert isinstance(index.chunk_storage, RAMChunkStorage)
    assert isinstance(index.chunking_algorithm, WordCountingChunkingAlgorithm)
    assert isinstance(index.embedder, BowEmbedder)
    assert isinstance(index.vector_index, L2RAMVectorIndex)

@pytest.mark.parametrize("index_class", MIRAGE_INDEX_IMPLEMENTATIONS)
def test_create_index(index_class, temp_data_folder):
    index = index_class(data_folder=temp_data_folder, words_amount_in_chunk=3)
    index.create_index()
    assert index.embedder.is_fitted == True
    assert index.vector_index.dim == index.embedder.get_dimensionality()

@pytest.mark.parametrize("index_class", MIRAGE_INDEX_IMPLEMENTATIONS)
def test_query(index_class, temp_data_folder):
    index = index_class(data_folder=temp_data_folder, words_amount_in_chunk=3)
    index.create_index()
    results = index.query("test", top_k=2, return_text=True)
    assert len(results) == 2
    assert any([
        "test" in result.text for result in results
    ])

@pytest.mark.parametrize("index_class", MIRAGE_INDEX_IMPLEMENTATIONS)
def test_custom_parameters(index_class, temp_data_folder):
    custom_raw_storage = FolderRawStorage(temp_data_folder)
    custom_chunk_storage = RAMChunkStorage()
    custom_chunking_algorithm = WordCountingChunkingAlgorithm(custom_raw_storage, custom_chunk_storage, words_amount=3)
    custom_embedder = BowEmbedder(normalizer=TextNormalizer())
    custom_vector_index = L2RAMVectorIndex(dimensionality=10)

    index = index_class(
        raw_storage=custom_raw_storage,
        chunk_storage=custom_chunk_storage,
        chunking_algorithm=custom_chunking_algorithm,
        embedder=custom_embedder,
        vector_index=custom_vector_index
    )
    assert index.raw_storage == custom_raw_storage
    assert index.chunk_storage == custom_chunk_storage
    assert index.chunking_algorithm == custom_chunking_algorithm
    assert index.embedder == custom_embedder
    assert index.vector_index == custom_vector_index

@pytest.mark.parametrize("index_class", MIRAGE_INDEX_IMPLEMENTATIONS)
def test_incorrect_custom_parameters(index_class):
    with pytest.raises(TypeError):
        # Некорректный тип для raw_storage
        index_class(raw_storage="invalid_type")

    with pytest.raises(TypeError):
        # Некорректный тип для chunk_storage
        index_class(chunk_storage="invalid_type")

    with pytest.raises(TypeError):
        # Некорректный тип для chunking_algorithm
        index_class(chunking_algorithm="invalid_type")

    with pytest.raises(TypeError):
        # Некорректный тип для embedder
        index_class(embedder="invalid_type")

    with pytest.raises(TypeError):
        # Некорректный тип для vector_index
        index_class( vector_index="invalid_type")