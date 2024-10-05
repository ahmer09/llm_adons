import os
import time
from pathlib import Path
from flask import Flask, request, jsonify, send_file, g
from langchain.storage import LocalFileStore
from langchain_core.prompts import ChatPromptTemplate
import configparser
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from tqdm import tqdm
from chunking import fixed_token_split, recursive_split
from ingestion import create_chroma_vector_store
from langchain_core.documents.base import Document
from retriever import naiveRetriever, multiQueryRetriever, contextualCompressionRetriever, ensembleRetriever
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from evaluation import RAGAS_withoutdata
from metadata import metadata_formatter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from embedding import generate_Azureembedding,generate_Transformerembedding
import pandas as pd
from azure.storage.blob import BlobServiceClient
import fitz
import datetime
from sqlalchemy import create_engine
import pymssql
import urllib

app = Flask(__name__)

@app.get("/")
def root():
    return jsonify({
        "app": "0.0.1v",
        "url": "https://modelopsappv2.azurewebsites.net/modelops"
    })

AZURE_OPENAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
AZURE_OPENAI_API_KEY = os.getenv('AZURE_OPENAI_API_KEY')
AZURE_OPENAI_API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
container_name = os.getenv('CONTAINER_NAME')
connection_string = os.getenv('CONTAINER_CONNECTION_STR')

SERVER = os.getenv('DB_SERVER')
DATABASE = os.getenv('DB_DATABASE')
USERNAME = os.getenv('DB_USERNAME')
PASSWORD = urllib.parse.quote_plus(os.getenv('DB_PASSWORD'))


ENDPOINT = os.getenv('OPENAI_ENDPOINT')
print(ENDPOINT)
api_key= os.getenv("OPENAI_API_KEY")
print(api_key)
API_VERSION = os.getenv('OPENAI_API_VERSION')
print(API_VERSION)

storage_path = './vectordb'

LANGCHAIN_TRACING_V2 = "true"
LANGCHAIN_API_KEY = os.getenv('LANGCHAIN_API_KEY')


def read_pdf_from_blob(connection_string,container_name,file_name):
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    blob_client = container_client.get_blob_client(file_name)

    try:
        download_stream = blob_client.download_blob()
        pdf_content = download_stream.readall()
        print(f"Length of PDF content: {len(pdf_content)} bytes")

        if not pdf_content:
            raise ValueError("The downloaded PDF content is empty.")

        pdf_document = fitz.open(stream=pdf_content, filetype="pdf")
        documents = [page.get_text() for page in pdf_document]

        if not documents:
            print("No text was extracted from the PDF.")

        nodes = [Document(page_content=text) for text in documents]
        print("returned nodes")
        return nodes    
    except Exception as e:
        print(f"Error reading PDF from Azure Blob Storage: {e}")
        return []
    return documents



def select_LLM(llm_option):
    if llm_option == "gpt35turbo":
        return "gpt35turbo"
    elif llm_option == "gpt4":
        return "gpt4"


def select_embedding(embedding):
    if embedding == "SentenceTransformer":
        option = generate_Transformerembedding()
        return option
    elif embedding == "ada0021_6":
        model = "ModelOps-text-embedding-ada-002"
        option = generate_Azureembedding(model)
        return option
    elif embedding == "text-embedding-3-large":
        model = "ModelOps-text-embedding-3-large"
        option = generate_Azureembedding(model)
        return option
    elif embedding == "text-embedding-3-small":
        model = "ModelOps-text-embedding-3-small"
        option = generate_Azureembedding(model)
        return option



def select_ingestion(ingest_option, embedding_function, splits):
    if ingest_option == "create_chroma_vector_store":
        suffix = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
        return create_chroma_vector_store(splits, embedding_function,suffix)

def select_retriever(retrieve_option, vector_store, embedding_function, llm,splits):
    if retrieve_option == "naiveRetriever":
        return naiveRetriever(db_storage_path=storage_path, embedding_function=embedding_function)
    elif retrieve_option == "multiQueryRetriever":
        return multiQueryRetriever(db_storage_path=storage_path, embedding_function=embedding_function, llm=llm)
    elif retrieve_option == "contextualCompressionRetriever":
        return contextualCompressionRetriever(db_storage_path=storage_path, embedding_function=embedding_function)
    elif retrieve_option == "ensembleRetriever":
        return ensembleRetriever(documents=splits, db_storage_path=storage_path, embedding_function=embedding_function, vectorstore=vector_store)
    


def select_chunking(chunk_option, docs, chunk_size, chunk_overlap):
    if chunk_option == "fixed_token_split":
        chunk = fixed_token_split(docs, chunk_size, chunk_overlap)
        return chunk
    elif chunk_option == "recursive_split":
        chunk= recursive_split(docs, chunk_size, chunk_overlap)
        return chunk
    elif chunk_option == "semantic_split":
        chunk = semantic_split(docs)
        return chunk 
    

def generate_metadata(docs, input):
    metadata = metadata_formatter(docs, input)
    if metadata:
        return metadata
    else:
        print("No metadata was generated")

def save_to_db(final_df):
    table_name = 'evaluation_results'
    print(USERNAME)
    print(PASSWORD)
    print(SERVER)
    print(DATABASE)
    connection_str = f"mssql+pymssql://{USERNAME}:{PASSWORD}@{SERVER}/{DATABASE}"
    print(connection_str)

    eng = create_engine(connection_str)

    with eng.connect() as connection:
        final_df.to_sql(table_name, connection, if_exists='append', index=False)
    
    print(f"Data saved to {DATABASE} in the {table_name} table.")


@app.before_request
def before_request():
    g.save_evaluation_df = pd.DataFrame()
    g.final_df = pd.DataFrame()

@app.post("/modelops")
def raguploader():
    chunking_options = ["fixed_token_split", "recursive_split"]
    ingestion_options = ["create_chroma_vector_store"]
    retriever_options = ["ensembleRetriever","multiQueryRetriever", "naiveRetriever"] 
    llm_options = ["gpt35turbo"]
    embedding_options = ["ada0021_6","text-embedding-3-large", "text-embedding-3-small","SentenceTransformer"]
    requestmode = request.json.get("RequestMode")
    print(requestmode)
    uploaded_file = request.json.get("FileName")
    print(uploaded_file)
    if uploaded_file:
        try:
            docs = read_pdf_from_blob(connection_string, container_name, uploaded_file)
        except Exception as e:
            return jsonify({"error": f"Error reading PDF: {str(e)}"}), 500
    else:
        return jsonify({"error": "No file uploaded"}), 400

    print("got nodes")
    print(docs)
    id_value = 0  

    chunk_option = request.json.get('ChunkingOption')
    print(chunk_option)
    ingest_option = request.json.get('IngestionOption')
    print(ingest_option)
    ret_option = request.json.get('RetrieverOption')
    print(ret_option)
    llm_option = request.json.get('LLMOption')
    print(llm_option)
    embed_option = request.json.get('EmbeddingOption')
    print(embed_option)

    compare_option = request.json.get('CompareOption')
    print(compare_option)


    total_iteration = 1
    if compare_option == "llm":
        compare_llm = compare_option
        total_iteration += len(llm_options)
    else:
        compare_llm = None    
    if compare_option == "chunk":
        compare_chunking = compare_option
        total_iteration += len(chunking_options)
    else:
        compare_chunking = None
    if compare_option == "embedding":
        compare_embedding = compare_option
        total_iteration += len(embedding_options)
    else:
        compare_embedding = None
    if compare_option == "ingestion":
        compare_ingestion = compare_option
        total_iteration += len(ingestion_options)
    else:
        compare_ingestion = None
    if compare_option == "retriver":
        compare_retriever = compare_option
        total_iteration += len(retriever_options)
    else:
        compare_retriever = None

    print(total_iteration)

    input = request.json.get('Question')
    print(input)
    ground_truth = request.json.get('GroundTruth')
    print(ground_truth)
        
    if input and requestmode == "Normal":
        system_prompt = (
            "You are an assistant for question-answering tasks. "
            "Use the following pieces of retrieved context to answer the question. "
            "If you don't know the answer, say that you don't know. "
            "Use three sentences maximum and keep the answer concise.\n\n{context}"
        )  

        id_value += 1
        id_value = f"Query_{id_value}"
        print(id_value)
        prompt = ChatPromptTemplate.from_messages([
                            ("system", system_prompt),
                            ("human", "{input}")
                                ])

        llm_options_to_run = llm_options if compare_llm else [llm_option]
        for llms in llm_options_to_run:
            print("------------------------------------------------------------------------------------------")
            print(llms)
            LLM_MODEL = select_LLM(llms)
            chunk_size = 1000
            chunk_overlap = int(0.15*chunk_size)
            chunk_options_to_run = chunking_options if compare_chunking else [chunk_option]
            for chunk_option in chunk_options_to_run:
                print("in chunking")
                print("------------------------------------------------------------------------------------------")
                print(chunk_option)
                splits = select_chunking(chunk_option, docs, chunk_size, chunk_overlap)
                print(f"Splits {splits}")
                embedding_options_to_run = embedding_options if compare_embedding else [embed_option]
                for embed_option in embedding_options_to_run:
                    print("------------------------------------------------------------------------------------------")
                    print("In embedding")
                    print(embed_option)
                    embedding_function = select_embedding(embed_option)
                    print(embedding_function)
                    ingestion_options_to_run = ingestion_options if compare_ingestion else [ingest_option]
                    for ingest_option in ingestion_options_to_run:
                        print("In ingestion")
                        print("------------------------------------------------------------------------------------------")
                        print(ingestion_options)
                        vector_store = select_ingestion(ingest_option, embedding_function, splits)
                        llm = AzureChatOpenAI(
                                    openai_api_type="azure",
                                    openai_api_version=API_VERSION,
                                    openai_api_key=api_key,
                                    azure_endpoint=ENDPOINT,
                                    deployment_name=LLM_MODEL,
                                    temperature=0
                                            )
                        retriever_options_to_run = retriever_options if compare_retriever else [ret_option]
                        for ret_option in retriever_options_to_run:
                            print("in retriver")
                            print("------------------------------------------------------------------------------------------")
                            print(ret_option)
                            retriever = select_retriever(ret_option, vector_store, embedding_function, llm, splits)
                            question_answer_chain = create_stuff_documents_chain(llm, prompt)
                            rag_chain = create_retrieval_chain(retriever, question_answer_chain)
                            total_iteration = total_iteration - 1
                            print(f"Total Iteration {total_iteration}")
                            response = rag_chain.invoke({"input": input})
                            print(response)
                            input_value = response['input']
                            question_rag = input_value
                            contexts = response.get('context')
                            answer = str(response.get('answer'))
                            configs = {
                                "Chunk Option": chunk_option,
                                "Ingestion Option": ingest_option,
                                "Retriever Option": ret_option,
                                "LLM Option": llm_option,
                                "Embedding Option": embed_option
                                    }

                            config_text = "  ".join([f"{key}: {value}" for key, value in configs.items()])
                            evaluation_df = RAGAS_withoutdata(docs, question_rag, answer, contexts,ground_truth, embedding_function)
                            if evaluation_df is not None:
                                evaluation_df = evaluation_df.dropna(how='all')
                                evaluation_df.insert(0, "Configuration", [config_text])
                                evaluation_df.insert(0, "ID", [id_value])
                                evaluation_df = evaluation_df.drop('contexts', axis=1)
                                new_column_names = ["ID","Configuration", "Question", "Answer", "Ground Truth", "Faithfulness", "Answer Relevancy", "Context Recall", "Context Precision"]
                                evaluation_df.columns = new_column_names
                                save_to_db(evaluation_df)
                                g.save_evaluation_df = pd.concat([g.save_evaluation_df, evaluation_df], ignore_index=True)
                                print(g.save_evaluation_df)
                                if total_iteration == 1: 
                                    print(total_iteration)
                                    g.final_df = g.save_evaluation_df
                                    print("FINAL DF")
                                    print(g.final_df)
                                    if g.final_df is not None:
                                        return jsonify({
                                            "message": "File processed successfully.",
                                            "evaluation": g.final_df.to_dict(orient='records')  
                                        }), 200
                                elif total_iteration == 0:
                                    g.final_df = evaluation_df
                                    print("FINAL DF")
                                    print(final_df)
                                    g.save_evaluation_df = evaluation_df
                                    if g.final_df is not None:
                                        return jsonify({
                                            "message": "File processed successfully.",
                                            "evaluation": g.final_df.to_dict(orient='records')  
                                        }), 200
    elif requestmode == "Metadata":
        metadata = generate_metadata(docs, input)
        print(metadata)  
        return metadata
return jsonify({
        "message": "Processing complete"
    }), 200   


if __name__ == "__main__":  
    app.run(host='0.0.0.0', port=5000)                    
