import os
import uuid
import chromadb
import faiss
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
import numpy as np 

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


def create_chroma_vector_store(docs):
    """
    creates chroma vector store
    :param docs: list of documents
    :return: chroma vector store
    """

    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embedding_function,
        client=chroma_client,
    )
    print(f"Added {len(docs)} chunks to chroma db")

    return vector_store

def create_faiss_vector_store(docs, embeddings):
    """
    creates faiss vector store
    :param docs: list of documents
    :return: faiss vector store
    """
    #serialized_docs = [doc.to_dict() for doc in docs]  # Assuming the Document class has a to_dict method.

    serialized_docs = [{
        "text": doc.page_content,
        "metadata": doc.metadata  # Adjust based on actual attributes of the Document class
    } for doc in docs]

    # Generate embeddings for the serialized documents
    embedding_results = embeddings.embed_documents([doc["text"] for doc in serialized_docs])
    embeddings_matrix = np.array(embedding_results) 

    print(embeddings_matrix)

    print("Embeddings Matrix Shape:", embeddings_matrix.shape)

    # Check if the embeddings matrix is not empty and has the correct dimensions
    if embeddings_matrix.shape[0] == 0 or len(embeddings_matrix.shape) < 2:
        raise ValueError("Embeddings matrix is empty or not formatted correctly.")

    # Initialize the FAISS index
    index = faiss.IndexFlatL2(embeddings_matrix.shape[1])

    # Add embeddings to the FAISS index
    index.add(embeddings_matrix)

    #index = faiss.IndexFlatL2(len(embeddings.embed_query(docs)))

    vector_store = FAISS.from_documents(
        documents=docs,
        embedding=embeddings,
        index=index
    )
    print(f"Added {len(docs)} chunks to faiss db")

    return vector_store
