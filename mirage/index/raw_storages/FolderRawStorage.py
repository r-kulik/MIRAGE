import os
import pdfplumber
import docx
import typing

from .RawStorage import RawStorage


class FolderRawStorage(RawStorage):
    """
    Basic Raw Storage that stores content of all text files in the folder 
    """

    def __init__(
            self, folder_path, 
            create_manually: bool = False, 
            extensions_of_text_files: tuple[str] = ('.txt', '.pdf', '.doc', '.docx'),
            custom_text_extractor: typing.Callable[[str], str] = None
        ) -> None:

        super().__init__()

        if folder_path is None:
            raise ValueError(
                "You are trying to create a FolderRawStorage object without specifying a folder"
            )

        self.__folder_path: str = folder_path
        self.__extensions_of_text_files = extensions_of_text_files
        
        self.__custom_text_extructor = custom_text_extractor

        if not create_manually:
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
        file_name: str = self._storage[index]
        if file_name.endswith('.pdf'):
            return FolderRawStorage.__read_pdf(file_name)
        if file_name.endswith(('.doc', '.docx')):
            return FolderRawStorage.__read_doc(file_name)
        if file_name.endswith('.txt'):
            return FolderRawStorage.__read_txt(file_name)
        return self.__custom_text_extructor(file_name)

    def __read_txt(file_name) -> str:
        with open(file_name, 'r', encoding='utf-8') as file:
            content = file.read()
        return content

    def __read_doc(file_name) -> str:
        doc = docx.Document(file_name)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        return '\n'.join(full_text)

    def __read_pdf(file_name) -> str:
        with pdfplumber.open(file_name) as pdf:
            full_text = []
            for page in pdf.pages:
                full_text.append(page.extract_text())
        return '\n'.join(full_text)