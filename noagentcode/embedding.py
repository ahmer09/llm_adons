from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv
from openai import OpenAI
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

load_dotenv()

ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
print(ENDPOINT)
api_key= os.getenv("AZURE_OPENAI_API_KEY")
print(api_key)
API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
print(API_VERSION)

def generate_Azureembedding(model):
    print(f"Initialized embeddings for model: {model}")
    embedding = AzureOpenAIEmbeddings(
        openai_api_key=api_key,
        azure_endpoint=ENDPOINT,
        openai_api_version=API_VERSION,
        azure_deployment=model,
        chunk_size=1
        )
    return embedding

def generate_Transformerembedding():
    embedding = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    return embedding

