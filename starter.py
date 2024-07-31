import os
import getpass
import tiktoken
import bs4
from dotenv import load_dotenv
import uuid

from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from rag.dataloader import DataLoader


# load api key
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

file_path = "./data/"
loaders = [
    TextLoader(file_path+"whatiworkedon.txt"),
    TextLoader(file_path+"thebestessay.txt"),
]
docs = []
for loader in loaders:
    docs.extend(loader.load())
#loader = DataLoader(file_path)
#docs = loader.lazy_load()


textsplitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = textsplitter.split_documents(docs)
print(len(splits))

#Embedding function
#default_ef = embedding_functions.DefaultEmbeddingFunction()
#embeddings = HuggingFaceEmbeddings()
embeddings = AzureOpenAIEmbeddings()

# vectorstore to use to index the child chunks
vectorstore = Chroma.from_documents(
    documents=splits, embedding_function=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 1})

docs = retriever.get_relevant_documents("what did the author do growing up?")
print(docs)