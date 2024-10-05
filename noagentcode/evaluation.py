from ragas import evaluate
from ragas.metrics import (faithfulness,answer_relevancy,context_recall,context_precision)
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import os
import asyncio
from langchain.vectorstores import Chroma
import pandas as pd
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from commoncode import llm, embedding
from langchain.schema import AIMessage




load_dotenv()

ENDPOINT = os.getenv('OPENAI_ENDPOINT')
API_KEY = os.getenv("OPENAI_API_KEY")
API_VERSION = os.getenv('OPENAI_API_VERSION')
MODEL = os.getenv('LLM_MODEL')
EMBEDDING = os.getenv('EMBEDDING_MODEL')
generator_llm = llm
critic_llm = llm

#Define metrics
metrics = [
        faithfulness,
        answer_relevancy,
        context_recall,  
        context_precision
    ]
    
def RAGAS_withoutdata(doc, questions, answers, context, ground_truth, embedding_function):
    print("__________________________________EVALUATION BY RAGAS WITH DATA_________________________________________________")

    if not ground_truth:
        llm_input = f"""Given the following document, generate a concise ground truth answer for the question provided. 
                        Question: {questions}

                        Document: {doc}

                        Please generate a relevant and accurate ground truth answer based on the content of the document."""
        #llm_input = f"Question: {questions}\nContext: {context}\nAnswer: {answers}"   
        ground_truth_response = llm.invoke(llm_input)
        ground_truth = ground_truth_response.content if isinstance(ground_truth_response, AIMessage) else str(ground_truth_response)
    else:
        ground_truth = ground_truth

    print(ground_truth)


    context_str = str(context)  # Ensure context is a string

    data = {
        "question": [questions],
        "answer": [answers],
        "contexts": [[context_str]],
        "ground_truth": [ground_truth]
    }

    
    #Convert dict to dataset
    dataset_ref = Dataset.from_dict(data)
   

    async def async_evaluate():
        return evaluate(
            dataset=dataset_ref, 
            metrics=metrics,
            llm=llm,
            embeddings=embedding_function
        )

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_evaluate())
    finally:
        loop.close()

    if result is not None:
        df = result.to_pandas()
        return df

    else:
        print("Evaluate returned None")
        return None
    

    


    