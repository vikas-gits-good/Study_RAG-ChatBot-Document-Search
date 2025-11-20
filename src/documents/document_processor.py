import os
import re
from pathlib import Path
from typing import List, Union, Literal
# from sentence_transformers import SentenceTransformer

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    TextLoader,
    WebBaseLoader,
    PyPDFLoader,
    PyPDFDirectoryLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_experimental.text_splitter import SemanticChunker

from ..config import OpenAIStuff


class DocumentProcessor:
    """Class to read, process data from different sources"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
    ) -> None:
        """Initialise document processor. Supports multiple
        document types.

        Args:
            chunk_size (int): Size of each text chunk/document. Defaults to 500.
            chunk_overlap (int): overlap between sequential chunk/document. Defaults to 50.
        """
        self.text_splt = RecursiveCharacterTextSplitter(
            separators=["\n\n\n", "\n\n", "\n", " ", ""],
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

    def load_from_url_links(self, urls: List[str]) -> List[Document]:
        """Method that takes in a list of urls (blogs) and return a single Document
        from each site.

        Args:
            urls (List[str]): Links to scrape text from.

        Returns:
            Documents (List[Document]): A list of Documents from each website.
        """
        urls = [urls] if isinstance(urls, str) else urls
        return [doc for url in urls for doc in WebBaseLoader(url).load()]

    def load_from_text_files(
        self,
        file_paths: Union[List[str], List[Path]],
        encoding: str = "utf-8",
    ) -> List[Document]:
        """Method that takes in a list of file paths to text files and returns a Document
        for each file.

        Args:
            file_paths (Union[List[str], List[Path]]): List of text file paths.
            encoding (str, optional): Text encoding. Defaults to "utf-8".

        Returns:
            Documents (List[Document]): A list of Documents from each text file.
        """
        paths = [file_paths] if isinstance(file_paths, str | Path) else file_paths
        return [doc for path in paths for doc in TextLoader(path, encoding).load()]

    def load_from_pdf_files(
        self,
        file_paths: Union[str, Path, List[str], List[Path]],
        return_type: Literal[
            "List[List[Document]]", "List[Document]"
        ] = "List[Document]",
    ) -> Union[List[List[Document]], List[Document]]:
        """Method that takes in a list of file paths to pdf files and returns a list
        of Document for each file. Each page in a file is a seperate Document.

        Args:
            file_paths (Union[str, Path, List[str], List[Path]]): File paths of pdf to be read.
            return_type (Literal[ &quot;List[List[Document]]&quot;, &quot;List[Document]&quot; ], optional): Return format of data. Defaults to "List[Document]".

        Returns:
            Documents (Union[List[List[Document]], List[Document]]): List of Document for each file.
        """
        paths = [file_paths] if isinstance(file_paths, str | Path) else file_paths
        return (
            [PyPDFLoader(path).load() for path in paths]
            if return_type == "List[List[Document]]"
            else [doc for path in paths for doc in PyPDFLoader(path).load()]
        )

    def load_pdf_from_directories(
        self,
        directories: Union[str, Path, List[str], List[Path]],
        return_type: Literal[
            "List[List[Document]]", "List[Document]"
        ] = "List[Document]",
    ) -> Union[List[List[Document]], List[Document]]:
        """Method that takes in a list of directories with pdf files and returns a list
        of Document for each file. Each page in a file is a seperate Document.

        Args:
            directories (Union[str, Path, List[str], List[Path]]): File paths of pdf to be read.
            return_type (Literal[ &quot;List[List[Document]]&quot;, &quot;List[Document]&quot; ], optional): Return format of data. Defaults to "List[Document]".

        Returns:
            Documents (Union[List[List[Document]], List[Document]]): List of Document for each file.
        """
        dirs = [directories] if isinstance(directories, str | Path) else directories
        return (
            [PyPDFDirectoryLoader(dir).load() for dir in dirs]
            if return_type == "List[List[Document]]"
            else [doc for dir in dirs for doc in PyPDFDirectoryLoader(dir).load()]
        )

    def separate_strings(self, strings):
        str_urls = []
        str_txts = []
        str_pdfs = []
        str_dirs = []

        url_pattern = re.compile(r"^https?://")  # regex for http or https at start

        for s in strings:
            if url_pattern.match(s):
                str_urls.append(s)
            elif s.endswith(".txt"):
                str_txts.append(s)
            elif s.endswith(".pdf"):
                str_pdfs.append(s)
            elif os.path.isdir(s):  # checks if s is an existing folder path
                str_dirs.append(s)
            else:
                raise ValueError(f"This item type '{s}' is not supported.")

        return str_urls, str_txts, str_pdfs, str_dirs

    def load_all_docs(self, sources: List[str]) -> List[Document]:
        """Method to load all the documents in supported format

        Args:
            sources (List[str]): Strings of file paths. directories or urls

        Returns:
            Documents (List[Document]): List of all Document with data.
        """
        sources = [sources] if isinstance(sources, str) else sources
        str_urls, str_txts, str_pdfs, str_dirs = self.separate_strings(sources)
        all_docs: List[Document] = []
        if str_urls:
            all_docs.extend(self.load_from_url_links(str_urls))
        if str_txts:
            all_docs.extend(self.load_from_text_files(str_txts))
        if str_pdfs:
            all_docs.extend(self.load_from_pdf_files(str_pdfs))
        if str_dirs:
            all_docs.extend(self.load_pdf_from_directories(str_dirs))
        return all_docs

    def chunk_all_docs(
        self,
        documents: List[Document],
        chunker: Literal["RCTS", "Semantic"] = "RCTS",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        separators: list[str] = ["\n\n\n", "\n\n", "\n", " ", ""],
    ) -> List[Document]:
        """Splits all the Documents into smaller parts using the seperators.

        Args:
            documents (List[Document]): List of Documents to be split.
            chunker (Literal[&quot;RCTS&quot;, &quot;Semantic&quot;], optional): Chunking Strategy. Defaults to "RCTS".
            chunk_size (int, optional): Approximate size of each chunk. Defaults to 500.
            chunk_overlap (int, optional): Overlap between consecutive chunks. Defaults to 50.
            separators (list[str], optional): Hierarchical order of seperator priority.

        Returns:
            Documents (List[Document]): List of split Documents.
        """
        chunker_dict = {
            "RCTS": RecursiveCharacterTextSplitter(
                separators=separators,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
            ),
            "Semantic": SemanticChunker(
                embeddings=OpenAIStuff().embedding,
                min_chunk_size=chunk_size,
            ),
        }
        chunker_obj = chunker_dict.get(chunker, "RCTS")
        return chunker_obj.split_documents(documents)

    def load_main(self, urls: List[str]) -> List[Document]:
        """Method to load and split documents.

        Args:
            urls (List[str]): List of urls to process.

        Returns:
            Documents (List[Document]): List of processed documents in chunks.
        """
        docs = self.load_all_docs(sources=urls)
        return self.chunk_all_docs(documents=docs)
