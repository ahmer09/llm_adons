import os
import time
from pathlib import Path
import streamlit as st
from langchain.storage import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate
import configparser
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from tqdm import tqdm
from chunking import fixed_token_split, recursive_split, semantic_split
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from ingestion import create_chroma_vector_store, create_faiss_vector_store
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_core.documents.base import Document
from retriever import naiveRetriever, multiQueryRetriever, contextualCompressionRetriever, ensembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from evaluation import RAGAS_withoutdata

# Read configuration file
config = configparser.ConfigParser()
config.read('config.ini')

# Azure OpenAI credentials
AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')


LLM_MODEL = "gpt35turbo"
EMBEDDING_MODEL = "ada0021_6"

COHERE_API_KEY = os.getenv('COHERE_API_KEY')
HF_TOKEN = os.getenv('HF_TOKEN')

#storage_path = config.get(section='database', option='storage_path')
#upload_path = config.get(section='database', option='upload_path')
storage_path = './vectordb'
upload_path = 'C:\\Users\\Hammer\\PycharmProjects\\llm_adons\\data\\upload'

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')

#embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
#embeddings=HuggingFaceInferenceAPIEmbeddings(
#    api_key=HF_TOKEN,
#    model_name='BAAI/bge-base-en-v1.5'
#)


# Create caching store for embeddings
store = LocalFileStore("./cache/")


chunking_options = ["fixed_token_split", "recursive_split", "semantic_split"]
ingestion_options = ["create_chroma_vector_store", "create_faiss_vector_store"]
retriever_options = ["naiveRetriever", "multiQueryRetriever", "contextualCompressionRetriever", "ensembleRetriever"]
llm_options = ["gpt35turbo", "gpt4"]
embedding_option = ["ada0021_6", "BAAI/bge-base-en-1.5", "ModelOps-text-embedding-3-large", "text-embedding-3-small"]

splits=[]

if __name__ == '__main__':

    # driver code
    st.set_page_config(page_title="RAG Uploader")
    with st.sidebar:
        text1 = st.sidebar.text("Choose a configuration:")
        chunk_option = st.sidebar.selectbox(label='Chunking option:', options = chunking_options)
        ingest_option = st.sidebar.selectbox(label='Ingestion option:', options = ingestion_options)
        retrieve_option = st.sidebar.selectbox(label='Retriever option:', options = retriever_options)
        llm_option = st.sidebar.selectbox(label='LLM option:', options = llm_options)
        embedding_option = st.sidebar.selectbox(label='Embedding option:', options = embedding_option)
    st.header("Upload pdf file to ask questions from.")
    prompt = st.chat_input("Say something")
    if prompt:
        st.write(f"User has sent the following prompt: {prompt}")

    # Get the file from file uploader
    uploaded_file = st.file_uploader("Upload a file")

    if uploaded_file is not None:
        save_path = Path(upload_path, uploaded_file.name)
        with open(save_path, mode='wb') as w:
            w.write(uploaded_file.getvalue())

        if save_path.exists():
            st.success(f'File {uploaded_file.name} is successfully saved!')

    if llm_option == "gpt35turbo":
        LLM_MODEL = "gpt35turbo"
    elif llm_option == "gpt4":
        LLM_MODEL = "gpt4"

    llm = AzureChatOpenAI(
        openai_api_type="azure",
        openai_api_version=AZURE_OPENAI_API_VERSION,
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        model=LLM_MODEL,
        temperature=0
    )

    if embedding_option == "ada0021_6":
        EMBEDDING_MODEL = "ada0021_6"
    elif embedding_option == "BAAI/bge-base-en-1.5":
        EMBEDDING_MODEL = "BAAI/bge-base-en-1.5"
    elif embedding_option == "ModelOps-text-embedding-3-large":
        EMBEDDING_MODEL = "ModelOps-text-embedding-3-large"
    elif embedding_option == "text-embedding-3-small":
        EMBEDDING_MODEL = "ModelOps-text-embedding-3-small"

    embeddings = AzureOpenAIEmbeddings(
        openai_api_type="azure",
        openai_api_version="2024-02-01",
        openai_api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        model=EMBEDDING_MODEL,
        allowed_special={'<|endoftext|>'}
    )
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings, store, namespace=embeddings.model
    )

    start = time.time()
    # Load document from upload folder
    loader = DirectoryLoader(upload_path, glob="*.pdf", loader_cls=PyPDFLoader)
    docs = loader.load()
    print('.....document_loaded.....')

    st.session_state['docs'] = docs

    if 'docs' in st.session_state:
        ## set chunk size
        chunk_size = 1000
        chunk_overlap = int(0.15*chunk_size)

        if chunk_option == "fixed_token_split":
            splits = fixed_token_split(docs, chunk_size, chunk_overlap)
        elif chunk_option == "recursive_split":
            splits = recursive_split(docs, chunk_size, chunk_overlap)
        elif chunk_option == "semantic_split":
            splits = semantic_split(docs)


        ## Ingest in vector chroma/FAISS
        if ingest_option == "create_chroma_vector_store":
            vector_store = create_chroma_vector_store(splits, cached_embeddings)
        else:
            vector_store = create_faiss_vector_store(splits, embeddings)

        ## call retriever method
        if retrieve_option == "naiveRetriever":
            retriever = naiveRetriever(vector_store)
        elif retrieve_option == "multiQueryRetriever":
            retriever = multiQueryRetriever(db_storage_path=storage_path, embedding_function=cached_embeddings, llm=llm)
        elif retrieve_option == "contextualCompressionRetriever":
            retriever = contextualCompressionRetriever(db_storage_path=storage_path, embedding_function=cached_embeddings)
        elif retrieve_option == "ensembleRetriever":
            retriever = ensembleRetriever(documents=splits, db_storage_path=storage_path, embedding_function=cached_embeddings, vectorstore=vector_store)

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
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        contexts = []
        submit = st.button("Ask the question")
        if submit:
            print("submitted.....")
            response = rag_chain.invoke({"input": input})
            input_value = response['input']
            question_rag = input_value
            print(question_rag)

            # Extracting context
            contexts = response.get('context')
            answer = response.get('answer')
            answer = str(answer)
            
            st.write(response['answer'])

            evaluation_df = RAGAS_withoutdata(question_rag, answer, contexts, docs)
            if evaluation_df is not None:
                st.write("Evaluation Results:")
                st.dataframe(evaluation_df) 
            else:
                st.write("No evaluation results to display.")

        stop = time.time()
        print("Total Time: ", stop - start)