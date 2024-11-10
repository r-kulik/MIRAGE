

from mirage.index.raw_storages.FolderRawStorage import FolderRawStorage

folder_storage = FolderRawStorage('data')
indexes = folder_storage.get_indexes()
print(
    folder_storage[indexes[0]]
)