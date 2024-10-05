import os
from typing import List

from dotenv import load_dotenv
from langchain.text_splitter import TokenTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain.text_splitter import Language, RecursiveCharacterTextSplitter

# load api key
load_dotenv()

def fixed_token_split(docs, chunk_size, chunk_overlap):
    """
    Fixed Token Chunking with overlap
    Args:
        :param docs: list of documents
        :param chunk_size: number of tokens
        :param chunk_overlap: token overlap between chunks
    :return:
        List[Document]: list of chunked documents
    """
    splitter = TokenTextSplitter(
        encoding_name="cl100k_base", chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)

"""def semantic_split(docs):
    'Semantic chunking
    Args:
        docs (List[Document]): List of documents to chunk
    Returns:
        List[Document]: List of chunked documents'
    splitter = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
    )
    return splitter.split_documents(docs)"""

def recursive_split(docs, chunk_size: int, chunk_overlap: int):
    """
    Recursive chunking

    Args:
        docs (List[Document]): List of documents to chunk
        chunk_size (int): Chunk size (number of tokens)
        chunk_overlap (int): Token overlap between chunks

    Returns:
        List[Document]: List of chunked documents
    """
    separators = ["\n\n", "\n", " ", ""]

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap

    )
    return splitter.split_documents(docs)