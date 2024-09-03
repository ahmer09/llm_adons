from langchain_openai import AzureOpenAIEmbeddings
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from openai import OpenAI
import os

load_dotenv()

ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
api_key= os.getenv("AZURE_OPENAI_API_KEY")
#print(API_KEY)
Openaikey= os.getenv("Openai_key")
API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
MODEL_new = os.getenv('LLM_MODEL')

ENDPOINT_new = os.getenv('AZURE_OPENAI_ENDPOINT_new')
api_key_new= os.getenv("AZURE_OPENAI_API_KEY_new")
API_VERSION_new = os.getenv('AZURE_OPENAI_API_VERSION_new')
MODEL_new = os.getenv('LLM_MODEL')

embedding_models = [
   'ada0021_6'
    #'ModelOps-text-embedding-3-large'
    #'text-embedding-3-small'

]

embedding_modelslarge = [
   #'ada0021_6'
    'ModelOps-text-embedding-3-large'
    #'text-embedding-3-small'

]

embedding_modelssmall = [
   #'ada0021_6'
    #'ModelOps-text-embedding-3-large'
    'text-embedding-3-small'

]

#client = OpenAI(api_key=Openaikey)


embeddings = []
def embedding_openai(doc):
        embedding = client.embeddings.create(
                input=doc,
                model="text-embedding-3-small"
        )
        embeddings.append((model,embedding))
        return embeddings

def embedding_azureopenai():
    embeddings = []
    for model in embedding_models:
        print(f"Initialized embeddings for model: {model}")
        embedding = AzureOpenAIEmbeddings(
        openai_api_key=api_key,
        azure_endpoint=ENDPOINT,
        openai_api_version=API_VERSION,
        azure_deployment=model,
        chunk_size=1
        )
        embeddings.append((model,embedding))
    return embeddings


def embedding_azureopenai_new():
    embeddings = []
    for model in embedding_modelslarge:
        print(f"Initialized embeddings for model: {model}")
        embedding = AzureOpenAIEmbeddings(
        openai_api_key=api_key_new,
        azure_endpoint=ENDPOINT_new,
        openai_api_version=API_VERSION_new,
        azure_deployment=model,
        chunk_size=1
        )
        embeddings.append((model,embedding))
    return embeddings

def embedding_azureopenai_small():
    embeddings = []
    for model in embedding_modelssmall:
        print(f"Initialized embeddings for model: {model}")
        embedding = AzureOpenAIEmbeddings(
        openai_api_key=api_key_new,
        azure_endpoint=ENDPOINT_new,
        openai_api_version=API_VERSION_new,
        azure_deployment=model,
        chunk_size=1
        )
        embeddings.append((model,embedding))
    return embeddings

def embedding_huggingface(documents):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    embeddings = []

    for doc in documents:
        # Assuming doc has a `page_content` attribute
        content = doc.page_content
        embedding = model.encode(content)
        embeddings.append((model, embedding))
    
    return embeddings
