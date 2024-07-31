from typing import Iterator, AsyncIterator
from langchain_core.document_loaders import BaseLoader
from langchain_core.documents import Document
import aiofiles

class DataLoader(BaseLoader):
    """creates a document loader that reads a file line by line"""
    def __init__(self, file_path: str) -> None:
        """Initialize the data loader with file_path

        Args:
            filepath: path to load the file
        """
        self.file_path = file_path

    def lazy_load(self) -> Iterator[Document]: # Does not take any arguments
        """A lazy loader that reads file line by line

        When you are implementing lazy load methods, you should a generator
        to yield documents one by one.
        """
        with open(self.file_path, encoding="utf-8") as file:
            line_number = 0
            for line in file:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path}
                )
                line_number += 1

    async def async_load(self) -> AsyncIterator[Document]:
        """ a async variant of lazy_load that reads file line by line"""

        async with aiofiles.open(self.file_path, encoding="utf-8") as file:
            line_number = 0
            async for line in file:
                yield Document(
                    page_content=line,
                    metadata={"line_number": line_number, "source": self.file_path}
                )
                line_number += 1


