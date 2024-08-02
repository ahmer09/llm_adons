import os
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import TokenTextSplitter

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings

# load api key
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

web_loader = WebBaseLoader(
    [
        "https://peps.python.org/pep-0483/",
        "https://peps.python.org/pep-0008/",
        "https://peps.python.org/pep-0257/",
    ]
)

pages = web_loader.load()

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

def semantic_split(docs):
    """
    Semantic chunking
    Args:
        docs (List[Document]): List of documents to chunk
    Returns:
        List[Document]: List of chunked documents
    """
    splitter = SemanticChunker(
        OpenAIEmbeddings(), breakpoint_threshold_type="percentile"
    )
    return splitter.split_documents(docs)

