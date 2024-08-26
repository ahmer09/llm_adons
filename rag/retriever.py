from langchain_chroma import Chroma
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.retrievers.weaviate_hybrid_search import WeaviateHybridSearchRetriever
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
    

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, 
        base_retriever=ensembleRetriever
    )

    return compression_retriever


def multiQueryRetriever(db_storage_path, embedding_function, llm):
    """
    Multi Query Retriever
    Args:
        @param db_storage_path: path to where the vectordb is saved in disk
        @param embedding_function: choice of embedding function used
        @param llm: choice of llm model used
    Returns:
        MultiQueryRetriever: object of multi query retriever
    """
    vectorstore = Chroma(
        embedding_function=embedding_function,
        persist_directory=db_storage_path
    )
    multi_query_retriever = MultiQueryRetriever.from_llm(
        retriever=vectorstore.as_retriever(), 
        llm=llm
    )
    
    return multi_query_retriever


def ensembleRetriever(documents, db_storage_path, embedding_function, vectorstore):
    """
    Ensemble Retriever
    Args:
        @param documents: input document which was loaded using documents loader
        @param db_storage_path: path to where the vectordb is saved in disk
        @param embedding_function: choice of embedding function used
    Returns:
        EnsembleRetriever: object of multi query retriever
    """
    bm25_retriever = BM25Retriever.from_documents(documents)
    bm25_retriever.k = 5

    #vectorstore = Chroma(
    #    embedding_function=embedding_function,
    #    persist_directory=db_storage_path
    #)
    retriever = vectorstore.as_retriever()

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, retriever], 
        weights=[0.5, 0.5]
    )
    
    return ensemble_retriever