import json
import openai
import re
import streamlit as st
from pathlib import Path
from langchain.document_loaders import DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from commoncode import llm

def extract_json(text):
    json_pattern = r'\{[^{}]*\}'
    matches = re.findall(json_pattern, text)
    json_objects = []
    for match in matches:
        try:
            json_object = json.loads(match)
            json_objects.append(json_object)
        except json.JSONDecodeError:
            continue  # Skip if it's not a valid JSON
    return json_objects

# Function to generate metadata using OpenAI
def metadata_formatter(document_info, user_queries):
    prompt = ('''You are a metadata expert with the task of analyzing document content to generate relevant metadata.

        Based on the provided document information, your goal is to create comprehensive and useful metadata. You are not required to follow a fixed format; instead, you should determine the most pertinent metadata fields based on the content you have. This can include, but is not limited to:

        - Summary or description of the document
        - Keywords or main topics
        - Author(s)
        - Publication date
        - Categories or tags
        - Document type
        - File size
        - Language
        - Estimated reading time
        - Significant sections or highlights

        Use your judgment to decide which fields are most relevant for the metadata based on the document's content. If the document information is not sufficient to determine certain fields, do not generate them or provide assumptions. Focus on providing accurate and valuable insights and information about the document.

        The metadata should be in JSON format.

        Here is the document information:
        {document_info}

        Provide your response in JSON format, including only the metadata fields that can be confidently derived from the document information.
        ''')

    main_prompt = [
        {"role": "system", "content": "You are an expert LLM Engineer and Consultant."},
        {"role": "user", "content": prompt}
    ]

    """try:
        #response = llm(messages=main_prompt)
        response = llm.invoke(main_prompt)
        #metadata_suggestion = response['choices'][0]['message']['content']
        print("Response received:", response)
        #metadata_suggestion = response.choices[0].message.content
        metadata_suggestion = response.content

        try:
            extracted_json = extract_json(metadata_suggestion)
            return extracted_json
        except json.JSONDecodeError:
            st.error("Failed to decode JSON. Here's the raw suggestion:")
            st.write(metadata_suggestion)

    except Exception as e:
        st.error(f"An error occurred: {e}")"""

    try:
        response = llm.invoke(main_prompt)
        metadata_suggestion = response.content
        metadata_json = json.loads(metadata_suggestion)
        return metadata_json
        #st.write("Metadata Suggestion:")
        #st.json(metadata_suggestion)  # Directly show the metadata suggestion

    except Exception as e:
        st.error(f"An error occurred: {e}")
