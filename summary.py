import os
import getpass
import tiktoken
from dotenv import load_dotenv

from langchain_chroma import Chroma
from langchain_core.stores import InMemoryStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader
from langchain.storage import InMemoryByteStore
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
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

