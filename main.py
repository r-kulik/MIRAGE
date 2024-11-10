

from mirage.index.raw_storages.FolderStorage import FolderStorage

folder_storage = FolderStorage('data')
indexes = folder_storage.get_indexes()
print(
    folder_storage[indexes[0]]
)