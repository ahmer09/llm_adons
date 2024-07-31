import uuid
import chromadb
from langchain_chroma import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dataloader import DataLoader
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
storage_path = "../vectordb"

chroma_client = chromadb.HttpClient(host=)
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")


collection = chroma_client.create_collection(name="test")

file_path = "../data/whatiworkedon.txt"


loader = PyPDFLoader("/data/phoenix.pdf")
pages = loader.load_and_split()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
chunked_documents = text_splitter.split_documents(pages)

Chroma.from_documents(
    documents=chunked_documents,
    embedding=embedding_function,
    collection_name="test",
    client=chroma_client,
)
print(f"Added {len(chunked_documents)} chunks to chroma db")

