from ragas import evaluate
from ragas.metrics import (faithfulness,answer_relevancy,context_recall,context_precision)
from datasets import Dataset
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings
import os
import asyncio
from langchain_chroma import Chroma
import pandas as pd
from ragas.testset.generator import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from commoncode import llm, embedding
from langchain.schema import AIMessage




load_dotenv()

ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT')
API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
API_VERSION = os.getenv('AZURE_OPENAI_API_VERSION')
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


def RAGA_withsytheticdata(doc, tests=10):
    print("__________________________________EVALUATION BY RAGAS WITH SYNTHETIC TEST DATA_________________________________________________")
    generator = TestsetGenerator.from_langchain(
    generator_llm,
    critic_llm,
    embedding
    )

    distributions = {
    simple: 0.5,
    multi_context: 0.4,
    reasoning: 0.1
    }
    
    testset = generator.generate_with_langchain_docs(doc, tests , distributions) 
    testset_df = testset.to_pandas()  
    print(testset_df.head()) 

    QUESTIONS_synthetic = testset_df['question'].to_list() 
    print("Synthetic Question", QUESTIONS_synthetic)
    GROUND_TRUTH_synthetic = testset_df['ground_truth'].to_list()  
    print("Synthetic Ground truth", GROUND_TRUTH_synthetic)

    eval_data = {
        "question": QUESTIONS_synthetic,
        "ground_truth": GROUND_TRUTH_synthetic,
        "contexts": [],
        "answer" :[]
    }

    #vectorstore = Chroma(embedding_function=embedding1)
    #retriever = vectorstore.as_retriever()

    for q in QUESTIONS_synthetic:
        eval_data["contexts"].append(
            [doc.page_content for doc in retriever.get_relevant_documents(q, k=3)]
        )

    for ques in QUESTIONS_synthetic:
        answer_synthetic = rag_chain.invoke(ques)
        eval_data["answer"].append(answer_synthetic)
        print(f"Generated answer for question '{ques}': {answer_synthetic}")


    data_synthetic = Dataset.from_dict(eval_data)

    async def async_evaluate():
        return evaluate(
            dataset=data_synthetic, 
            metrics=metrics,
            llm=llm,
            embeddings=embedding
        )

    #loop = asyncio.new_event_loop()
    #asyncio.set_event_loop(loop)
    """loop = asyncio.get_event_loop()
    result = loop.run_until_complete(async_evaluate())
    loop.close()"""

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_evaluate())
    finally:
        loop.close()

    if result is not None:
        df = result.to_pandas()
        df.to_excel('./raga_output.xlsx', index=False, engine='openpyxl')
        print(df)

    else:
        print("Evaluate returned None")

    

    
def RAGAS_withoutdata(questions, answers, context, doc):
    print("__________________________________EVALUATION BY RAGAS WITH DATA_________________________________________________")

    # Generate ground truth using the LLM
    llm_input = f"Given the context below, generate a response for the question:\nQuestion: {questions}\nContext: {context}\nGenerate a response."

    #llm_input = f"Question: {questions}\nContext: {context}\nAnswer: {answers}"
    
    ground_truth_response = llm.invoke(llm_input)

    ground_truth = ground_truth_response.content if isinstance(ground_truth_response, AIMessage) else str(ground_truth_response)

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
            embeddings=embedding
        )

    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_evaluate())
    finally:
        loop.close()

    if result is not None:
        df = result.to_pandas()
        df.to_excel('./raga_output.xlsx', index=False, engine='openpyxl')
        print(df)
        return df

    else:
        print("Evaluate returned None")
        return None
    

    


    