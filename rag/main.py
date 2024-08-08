import chromadb
from langchain_chroma import Chroma
from datasets import load_dataset
from tqdm import tqdm
from rag.chunking import fixed_token_split, recursive_split
from rag.ingestion import create_vector_store
from rag.ragas_evaluate import perform_evaluation
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents.base import Document

from rag.ragas_evaluate import perform_evaluation

if __name__ == '__main__':

    loader = DirectoryLoader("../data/", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print('.....document_loaded.....')
    chunk_size = 1000
    chunk_overlap = int(0.15*chunk_size)
    print(f"CHUNK SIZE: {chunk_size}")
    print("------ Fixed token with overlap ------")

    splits = fixed_token_split(docs, chunk_size, chunk_overlap)
    vector_store = create_vector_store(splits)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    result = perform_evaluation(docs, retriever)
    print(
        f"Result: {result}"
    )