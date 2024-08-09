import os
from pathlib import Path
import streamlit as st
from werkzeug.utils import secure_filename
import chromadb
from langchain_chroma import Chroma
from datasets import load_dataset
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from tqdm import tqdm
from chunking import fixed_token_split, recursive_split
from ingestion import create_vector_store
from ragas_evaluate import perform_evaluation
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents.base import Document
from ragas_evaluate import perform_evaluation
from retriever import naiveRetriever

storage_path = "../vectordb"

chroma_client = chromadb.PersistentClient(storage_path)
collection = "test"
chroma_vector_search_index_name = "vector_index"
EMBEDDING_MODEL = "ada0021_6"



AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_version="2024-02-01",
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model=EMBEDDING_MODEL,
    allowed_special={'<|endoftext|>'}
)


if __name__ == '__main__':

    # driver code
    st.set_page_config(page_title="RAG Uploader")
    st.header("Upload pdf file to ask questions from.")

    # Get the file from file uploader
    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file is not None:
        save_folder = 'C:\\Users\\Hammer\\PycharmProjects\\llm_adons\\data\\upload'
        save_path = Path(save_folder, uploaded_file.name)
        with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())

        if save_path.exists():
            st.success(f'File {uploaded_file.name} is successfully saved!')

    loader = DirectoryLoader("C:\\Users\\Hammer\\PycharmProjects\\llm_adons\\data\\upload", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print('.....document_loaded.....')
    chunk_size = 1000
    chunk_overlap = int(0.15*chunk_size)

    splits = fixed_token_split(docs, chunk_size, chunk_overlap)
    vector_store = create_vector_store(splits)
    retriever = naiveRetriever(storage_path, azure_embeddings)

    # Take user question..
    input = st.text_input("Input: ", key="input")

    submit = st.button("Ask the question")
    if submit:
        print("submitted.....")
        retrieved_docs = retriever.invoke(input)
        print(f"\n{'-' * 100}\n".join([f"Document {i + 1}:\n" + d.page_content for i, d in enumerate(retrieved_docs)]))


    #result = perform_evaluation(docs, retriever)
    #print(
    #    f"Result: {result}"
    #)