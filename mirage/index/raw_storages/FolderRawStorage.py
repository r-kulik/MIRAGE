import json
import os
import zipfile
from loguru import logger
import pdfplumber
import docx
import typing

from tqdm import tqdm

from .RawStorage import RawStorage


class FolderRawStorage(RawStorage):
    """
    Basic Raw Storage that stores content of all text files in the folder 
    """
    CONFIG_FILENAME: str = 'folder_raw_storage.mirage_config'

    def __init__(
            self, folder_path, 
            create_manually: bool = False, 
            extensions_of_text_files: tuple[str] = ('.txt', '.pdf', '.doc', '.docx'),
            custom_text_extractor: typing.Callable[[str], str] = None
        ) -> None:

        super().__init__()

        self.__folder_path: str | None = folder_path
        self.__extensions_of_text_files = extensions_of_text_files
        
        self.__custom_text_extructor = custom_text_extractor

        if not create_manually and self.__folder_path is not None:
            self.__createStorage()
        
    def __createStorage(self):
        file_list: list[str] = os.listdir(self.__folder_path)
        for file in file_list:
            if file.endswith(self.__extensions_of_text_files):
                self.add_to_storage(file, os.path.join(self.__folder_path, file))

    
    
    def __getitem__(self, index: str) -> str:
        """
        :param: index in the storage
        :returns: text of the document
        """
        if self.__folder_path is None:
            raise RuntimeError(
                "You are trying to obtain data from FolderRawStorage without specified folder"
            )

        file_name: str = self._storage[index]
        if file_name.endswith('.pdf'):
            return FolderRawStorage.__read_pdf(file_name)
        if file_name.endswith(('.doc', '.docx')):
            return FolderRawStorage.__read_doc(file_name)
        if file_name.endswith('.txt'):
            return FolderRawStorage.__read_txt(file_name)
        return self.__custom_text_extructor(file_name)

    def __read_txt(file_name) -> str:
        try:
            with open(file_name, 'r', encoding='utf-8') as file:
                content = file.read()
            return content
        except UnicodeDecodeError:
            import chardet
            with open(file_name, 'rb') as f:
                raw_data = f.read()
                result = chardet.detect(raw_data[:10000])
                encoding = result['encoding']
                return raw_data.decode(encoding)
            

    def __read_doc(file_name) -> str:
        doc = docx.Document(file_name)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)

    def __read_pdf(file_name) -> str:
        with pdfplumber.open(file_name) as pdf:
            full_text = []
            logger.info(f'Reading pages of {file_name}')
            for page in tqdm(pdf.pages):
                full_text.append(page.extract_text())
        return '\n'.join(full_text)
    
    def save(self, filename_to_save: str) -> None:
        """Saves a whole folder to a zip file

        Parameters
        ----------
        filename : str
            Name of the file to store FolderRawStorage in
        """

        if self.__custom_text_extructor is not None:
            raise Warning(
                "Currently saving FolderRawStorages with custom text extractors is not supported. \
                    Hovewer, it is not necessary for correct operation of the stored index, if there would not be further training"
            )
        info_json = {
            "original_folder_path": self.__folder_path,
            "extensions_of_text_files": list(self.__extensions_of_text_files),
            "custom_text_extractor": self.__custom_text_extructor is not None,
            "storage": self._storage
        }
        with open(FolderRawStorage.CONFIG_FILENAME, 'w', encoding='utf-8') as infofile:
            infofile.write(json.dumps(info_json))
        
        with zipfile.ZipFile(filename_to_save, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for filename in self._storage:
                file_path = self._storage[filename]
                zipf.write(file_path, filename)
            
            zipf.write(FolderRawStorage.CONFIG_FILENAME, FolderRawStorage.CONFIG_FILENAME)
        os.remove(FolderRawStorage.CONFIG_FILENAME)
    
    @staticmethod
    def load(filename: str) -> typing.Self:
        with zipfile.ZipFile(filename, 'r') as zipf:
            if not FolderRawStorage.CONFIG_FILENAME in zipf.namelist():
                raise ValueError(
                    f"You are trying to restore a FolderRawStorage from the file {filename} which does not meet a requirements or it is corrupted"
                )
            zipf.extract(FolderRawStorage.CONFIG_FILENAME, '')
            with open(FolderRawStorage.CONFIG_FILENAME, 'r', encoding='utf-8') as infofile:
                info_json = json.loads(infofile.read())
            folder_for_extraction_path = info_json['original_folder_path'] + "_restored"
            for filename in info_json['storage']:
                if not filename in zipf.namelist():
                    raise ValueError(
                        "Configuration info of the folder storage is linkning to the file that is not presented in the archive because of the corruption of the archive"
                    )
                zipf.extract(filename, folder_for_extraction_path)
            
            
        folder_storage_to_return = FolderRawStorage(
            folder_path=info_json['original_folder_path'] + "_restored",
            extensions_of_text_files=tuple(info_json["extensions_of_text_files"]),
            custom_text_extractor=None,
            create_manually=True
        )
        for filename in info_json['storage']:
            folder_storage_to_return._storage[filename] = os.path.join(
                folder_for_extraction_path, filename
            )
        os.remove(FolderRawStorage.CONFIG_FILENAME)
        return folder_storage_to_return