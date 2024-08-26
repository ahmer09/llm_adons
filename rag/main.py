import os
import time
from pathlib import Path
import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from werkzeug.utils import secure_filename
import chromadb
from langchain_chroma import Chroma
from datasets import load_dataset
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from tqdm import tqdm
from chunking import fixed_token_split, recursive_split
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.vectorstores import FAISS
from ingestion import create_chroma_vector_store, create_faiss_vector_store
from ragas_evaluate import perform_evaluation
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents.base import Document
from ragas_evaluate import perform_evaluation
from retriever import naiveRetriever, multiQueryRetriever, contextualCompressionRetriever, ensembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT = "https://azureopenai16.openai.azure.com/"
AZURE_OPENAI_API_KEY = "75db73a3b9da40b0b6e0e98273a6029f"
AZURE_OPENAI_API_VERSION = "2024-02-01"

LLM_MODEL = "gpt35turbo"
EMBEDDING_MODEL = "ada0021_6"

COHERE_API_KEY = "BWR8YyveaadqsWa8Ty0FM0vEIAysgJgnjhZVkRP1"
HF_TOKEN = "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"

storage_path = "./vectordb"

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = "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"

#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
embeddings=HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_TOKEN,
    model_name='BAAI/bge-base-en-v1.5'
)

print(embeddings)

llm = AzureChatOpenAI(
  openai_api_type="azure",
  openai_api_version="2024-02-01",
  openai_api_key=AZURE_OPENAI_API_KEY,
  azure_endpoint=AZURE_OPENAI_ENDPOINT,
  model=LLM_MODEL,
  temperature=0
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

    start = time.time()
    # Load document from upload folder
    loader = DirectoryLoader("C:\\Users\\Hammer\\PycharmProjects\\llm_adons\\data\\upload", glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print('.....document_loaded.....')

    ## set chunk size
    chunk_size = 1000
    chunk_overlap = int(0.15*chunk_size)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    splits = splitter.split_documents(docs)
    print(type(splits))

    ## store in vector chroma/FAISS
    #vector_store = create_chroma_vector_store(splits)
    vector_store = create_faiss_vector_store(splits, embeddings=embeddings)

    ## call retriever method
    #retriever = multiQueryRetriever(db_storage_path=storage_path, embedding_function=embedding_function, llm=llm)
    ensemble_retriever = ensembleRetriever(documents=splits, db_storage_path=storage_path, embedding_function=embeddings, vectorstore=vector_store)
    compression_retriever  = contextualCompressionRetriever(db_storage_path=storage_path, embedding_function=embeddings)

    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know. Use three sentences maximum and keep the "
        "answer concise."
        "\n\n"
        "{context}"
    )

    # Take user question..
    input = st.text_input("Input: ", key="input")

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(ensemble_retriever, question_answer_chain)

    submit = st.button("Ask the question")
    if submit:
        print("submitted.....")
        response = rag_chain.invoke({"input": input})
        #retrieved_docs = retriever.get_relevant_documents(input)
        st.write(response['answer'])

    stop = time.time()
    print("Total Time: ", stop - start)