import json
import openai
import re
from pathlib import Path
from langchain.text_splitter import RecursiveCharacterTextSplitter
import time
from commoncode import llm
from flask import jsonify  

def metadata_formatter(docs, input):
    response = []
    metadata_suggestion = []

    docs_str = str(docs)

    prompt = (f'''Your task is to extract and present document information, keywords, and context relevant to the input '{input}' from the provided document. Identify and format the content clearly, ensuring the extracted information is concise and related to the input. Present the keywords derived from the content and provide a brief contextual summary.

                    Here is the document: {docs_str}

                    If you are unable to find specific information or keywords related to the input, return the most relevant portions of the original document.

                    Format the output as follows:

                    Context: [Brief summary or context derived from the document related to the input]
                    Keywords: [List of keywords derived from the document]
            ''')
    main_prompt = [
        {"role": "system", "content": "You are an expert LLM Engineer and Consultant."},
        {"role": "user", "content": prompt}
    ]

    try:
        # Invoke the LLM with the prompt
        response = llm.invoke(main_prompt)
        print(f"Response : {response}")
        if hasattr(response, 'content'):
            metadata_suggestion = response.content  
        else:
            metadata_suggestion = str(response)  

        print(f"Response : {response}")
        print(f"Metadata Suggestion: {metadata_suggestion}")

        
        return jsonify({"metadata": metadata_suggestion})

    except Exception as e:
        print(f"An error occurred: {e}")
        return jsonify({"error": "Could not jsonify metadata", "details": str(e)}), 500  