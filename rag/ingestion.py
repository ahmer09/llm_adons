import os
import uuid
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai.embeddings import AzureOpenAIEmbeddings

storage_path = "../vectordb"

chroma_client = chromadb.PersistentClient(storage_path)
collection = "test"
chroma_vector_search_index_name = "vector_index"
EMBEDDING_MODEL = "ada0021_6"



AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#azure_embeddings = AzureOpenAIEmbeddings(
#    openai_api_type="azure",
#    openai_api_version="2024-02-01",
#    openai_api_key=AZURE_OPENAI_API_KEY,
#    azure_endpoint=AZURE_OPENAI_ENDPOINT,
#    model=EMBEDDING_MODEL,
#    allowed_special={'<|endoftext|>'}
#)


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
