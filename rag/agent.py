import os
import time
from pathlib import Path
import streamlit as st
import configparser
from dotenv import load_dotenv
load_dotenv()
from llama_index.core.agent import ReActAgent
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
import streamlit as st
from streamlit_extras.stylable_container import stylable_container
from langchain.agents import AgentExecutor, create_openai_tools_agent

chunking_options = ["fixed_token_split", "recursive_split", "semantic_split"]
ingestion_options = ["create_chroma_vector_store", "create_faiss_vector_store"]
retriever_options = ["ensembleRetriever","naiveRetriever", "multiQueryRetriever", "contextualCompressionRetriever"]
llm_options = ["gpt35turbo", "gpt4"]
embedding_option = ["ada0021_6","SentenceTransformer", "BAAI/bge-base-en-1.5", "ModelOps-text-embedding-3-large", "text-embedding-3-small"]

#config = configparser.ConfigParser.read("")
#upload_path = config.get(section='upload', option='upload_path')
upload_path = 'C:\\Users\\Hammer\\PycharmProjects\\llm_adons\\data\\upload'

def trigger_agent(prompt):

    #llm = Ollama(model="mixtral:8x7b", request_timeout=120.0)
    llm = OpenAI(model="gpt-3.5-turbo", temperature=0)

    documents = SimpleDirectoryReader(upload_path).load_data()
    index = VectorStoreIndex(documents)
    query_engine = index.as_query_engine()
    answer = query_engine.query(prompt)

    return answer



if __name__ == '__main__':

    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        with open(save_path, mode="wb") as f:
            f.write(uploaded_file.read())

        if save_path.exists():
            st.success(f'File {uploaded_file.name} has been uploaded.')

        # Initialize chat history
        if "openai_model" not in st.session_state:
            st.session_state["openai_model"] = "gpt-3.5-turbo"

        if 'messages' not in st.session_state:
            st.session_state['messages'] = []

        # Display chat history on app run
        for message in st.session_state['messages']:
            with st.chat_message(message['role']):
                st.markdown(message['content'])

        with stylable_container(
            key="bottom_container",
            css_styles="""
                .{
                    position: fixed;
                    bottom: 120px;
                }
            """,
        ):
            st.button("Show Evaluation Metric")

        if prompt := st.chat_input("Ask me!"):
            # Display user message in chat message container
            st.chat_message("user").markdown(prompt)
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})

            response = trigger_agent(prompt)
            bot_response = f"{response}"
            # Display bot response in chat message container
            with st.chat_message("assistant"):
                st.markdown(bot_response)
            # Add bot response to chat history
            st.session_state.messages.append({"role": "assistant", "content": bot_response})




