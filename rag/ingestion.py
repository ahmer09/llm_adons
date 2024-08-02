import uuid
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataloader import DataLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

storage_path = "../vectordb"

chroma_client = chromadb.PersistentClient(storage_path)
collection = "test"
chroma_vector_search_index_name="vector_index"
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def create_vector_store(docs):
    """
    creates chroma vector store
    :param docs: list of documents
    :return: chroma vector store
    """
    vector_store = Chroma.from_documents(
    documents=docs,
    embedding=embedding_function,
    collection_name="test",
    client=chroma_client,
)
    print(f"Added {len(docs)} chunks to chroma db")

    return vector_store

