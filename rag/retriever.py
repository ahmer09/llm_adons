from langchain_chroma import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
import os


AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
LLM_MODEL = os.getenv("LLM_MODEL")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL")


def naiveRetriever(db_storage_path, embedding_function, k=3):
    """
    Naive Retriever
    Args:
        @param db_storage_path: path to where the vectordb is saved in disk
        @param embedding_function: choice of embedding function used
        @param k: top k relevant documents to be retrieved
    Returns:
        VectorStoreRetriever: object of vectorstore as a retriever
    """
    vector_db = Chroma(
        persist_directory=db_storage_path, 
        embedding_function=embedding_function
    )
    retriever = vector_db.as_retriever(search_kwargs={"k" : k})
    
    return retriever
    

def contextualCompressionRetriever(db_storage_path, embedding_function, k=3):
    """
    Contextual Compression Retriever(Reranking)
    Args:
        @param db_storage_path: path to where the vectordb is saved in disk
        @param embedding_function: choice of embedding function used
        @param k: top k relevant documents to be retrieved
    Returns:
        ContexualCompressionRetriever: object of contextual compression retriever
    """
    os.environ["COHERE_API_KEY"] = os.getenv('COHERE_API_KEY')
    
    compressor = CohereRerank(
        top_n=k,
        model="rerank-english-v3.0"
    )
    
    naive_retriever = naiveRetriever(
        db_storage_path=db_storage_path, 
        embedding_function=embedding_function, 
        k=k+5
    )
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=naive_retriever
    )

    return compression_retriever