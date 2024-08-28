import os
import bs4
from dotenv import load_dotenv
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain import hub
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
#print(API_KEY)
API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
MODEL = os.getenv('LLM_MODEL')
EMBEDDING = os.getenv('EMBEDDING_MODEL')


embedding = AzureOpenAIEmbeddings(
    openai_api_key=API_KEY,
    azure_endpoint=ENDPOINT,
    openai_api_version=API_VERSION,
    azure_deployment=EMBEDDING,
    chunk_size=1
)

llm = AzureChatOpenAI(
            openai_api_type="azure",
            openai_api_version=API_VERSION,
            openai_api_key=API_KEY,
            azure_endpoint=ENDPOINT,
            deployment_name=MODEL,
            temperature=0
        )

#vectorstore = Chroma(embedding_function=embedding)

#retriever = vectorstore.as_retriever()

#prompt_hub_rag = hub.pull("rlm/rag-prompt")
"""system_prompt = (
                "You are an assistant for question-answering tasks. "
                "Use the following pieces of retrieved context to answer the question."
                "If you don't know the answer, just say that you don't know."
                "Use three sentences maximum and keep the answer concise."
                "Question: {question}\n"
                "Context: {context}\n"

                 "Answer:"
            )


ragchain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | system_prompt
    | llm
    | StrOutputParser()
)"""
