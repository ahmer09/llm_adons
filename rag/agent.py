import os
import time
from pathlib import Path
import streamlit as st
import configparser
from dotenv import load_dotenv
from llama_index.core.agent import ReActAgent
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from langchain.agents import AgentExecutor, create_openai_tools_agent
from commoncode import llm
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
from evaluation import RAGAS_withoutdata
from metadata import metadata_formatter
import pandas as pd

load_dotenv()

ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
#print(API_KEY)
API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
MODEL = os.getenv('LLM_MODEL')
EMBEDDING = os.getenv('EMBEDDING_MODEL')




chunking_options = ["fixed_token_split", "recursive_split", "semantic_split"]
ingestion_options = ["create_chroma_vector_store", "create_faiss_vector_store"]
retriever_options = ["ensembleRetriever","naiveRetriever", "multiQueryRetriever", "contextualCompressionRetriever"]
llm_options = ["gpt35turbo", "gpt4"]
embedding_option = ["ada0021_6","SentenceTransformer", "BAAI/bge-base-en-1.5", "ModelOps-text-embedding-3-large", "text-embedding-3-small"]

#config = configparser.ConfigParser.read("")
#upload_path = config.get(section='upload', option='upload_path')
upload_path = 'C:\\Users\\Apnavi\\Desktop\\Code\\New clone\\llm_adons\\data\\upload'

llm=None

def trigger_agent(prompt):
    azurellm = AzureChatOpenAI(
        openai_api_type="azure",
        openai_api_version=API_VERSION,
        openai_api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        deployment_name=MODEL,
        temperature=0
    )

    Settings.llm = azurellm

    
    documents = SimpleDirectoryReader(upload_path).load_data()
  
    embedding = AzureOpenAIEmbeddings(
        openai_api_key=API_KEY,
        azure_endpoint=ENDPOINT,
        openai_api_version=API_VERSION,
        azure_deployment=EMBEDDING,
        chunk_size=1
    )
    index = VectorStoreIndex(documents, embed_model=embedding, llm=azurellm)
    query_engine = index.as_query_engine()
    answer = query_engine.query(prompt)
    print(answer)
    
    return {
        "answer": answer.response,
        "context": [node.node.get_text() for node in answer.source_nodes]  
    }

def generate_metadata(input_text, context):
    user_queries = [input_text] 
    metadata = metadata_formatter(context, user_queries)  
    return metadata


def show_evaluation(question_rag, answer, contexts, docs):
    evaluation_df = RAGAS_withoutdata(question_rag, answer, contexts, docs)
    return evaluation_df




if __name__ == '__main__':
    st.set_page_config(page_title="ragbot")
    with st.sidebar:
        text1 = st.sidebar.text("Choose a configuration:")
        chunk_option = st.sidebar.selectbox(label='Chunking option:', options=chunking_options)
        ingest_option = st.sidebar.selectbox(label='Ingestion option:', options=ingestion_options)
        retrieve_option = st.sidebar.selectbox(label='Retriever option:', options=retriever_options)
        llm_option = st.sidebar.selectbox(label='LLM option:', options=llm_options)
        embedding_option = st.sidebar.selectbox(label='Embedding option:', options=embedding_option)
    st.header("Upload pdf file to ask questions from.")

    uploaded_file = st.file_uploader("Choose a pdf file", type="pdf")
    if uploaded_file is not None:
        save_path = Path(upload_path, uploaded_file.name)
        print(f"Save path is {save_path}")
        with open(save_path, mode="wb") as f:
            f.write(uploaded_file.read())

        if save_path.exists():
            st.success(f'File {uploaded_file.name} has been uploaded.')

        st.session_state['last_uploaded_file'] = save_path

        # Initialize chat history
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        # Display chat history on app run
        for message in st.session_state['messages']:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        if prompt := st.chat_input("Ask me!"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = trigger_agent(prompt)
            st.session_state['last_prompt'] = prompt  # Save the last prompt
            st.session_state['last_response'] = response  # Save the last response

            bot_response = f"{response['answer']}"
            # Display bot response in chat message container
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
        response_container = st.container()

        with response_container:
            if st.session_state.get('show_eval_metric', False): 
                if 'last_prompt' in st.session_state and 'last_response' in st.session_state:
                    question_rag = st.session_state['last_prompt']
                    answer = st.session_state['last_response']['answer']
                    contexts = st.session_state['last_response']['context']
                    docs = SimpleDirectoryReader(upload_path).load_data()  

                    configs = {
                        "Chunk Option": chunk_option,
                        "Ingestion Option": ingest_option,
                        "Retriever Option": retrieve_option,
                        "LLM Option": llm_option,
                        "Embedding Option": embedding_option
                    }
                    config_text = "\n".join([f"{key}: {value}" for key, value in configs.items()])

                    evaluation_df = show_evaluation(question_rag, answer, contexts, docs)

                    if not evaluation_df.empty:
                        evaluation_df = evaluation_df.dropna(how='all')  

                        
                        evaluation_df.insert(0,"Configuration", [config_text] + [''] * (len(evaluation_df) - 1))

                        st.chat_message("assistant").markdown("Selected Configuration & Evaluation Results:")
                        st.dataframe(evaluation_df)  

                        st.session_state['evaluation_df'] = evaluation_df
                    else:
                        st.write("No evaluation results to display.")

            with st.container():
                st.markdown(
                    """
                    <style>
                    .button-container {
                        position: fixed;
                        bottom: 20px;  /* Adjust position for both buttons */
                        left: 50%;  /* Center align the container */
                        transform: translateX(-50%);  /* Correct centering with translate */
                        width: auto;  /* Auto width to fit content */
                        display: flex;
                        justify-content: space-between;  /* Space between buttons */
                        z-index: 1000; /* Ensure buttons are above other content */
                    }
                    .button-container .stButton {
                        margin: 10px;  /* Space between buttons */
                    }
                    </style>
                    """,
                    unsafe_allow_html=True
                )

                button_container = st.container()
                with button_container:
                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("Show Evaluation Metric", key="eval_metric_button_2"):
                            st.session_state['show_eval_metric'] = True
                