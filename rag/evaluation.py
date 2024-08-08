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
from commoncode import llm, embedding, rag_chain, vectorstore, retriever



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
    questions = questions[0].strip().lower()
    print("__________________________________EVALUATION BY RAGAS WITH DATA_________________________________________________")
    eval_questions = [
        "What are the essential pre-processing steps involved in training Large Language Models (LLMs)?",
        "How do positional encodings function in transformers, and why are they important for LLMs?",
        "What are some of the emergent abilities of LLMs that have contributed to their wide adoption?",
        "What are the main challenges associated with the training and inference of Large Language Models?",
        "What role does attention play in LLMs, and what are some of the different types of attention used?"
    ]

    ground_truths = [
        ["Tokenization is an essential pre-processing step in LLM training, where text is parsed into non-decomposing units called tokens. Common tokenization schemes used in LLMs include WordPiece, Byte Pair Encoding (BPE), and UnigramLM."],
        ["In transformers, positional encodings are added to token embeddings to provide positional information, which the transformer architecture inherently lacks. This allows the model to capture the order of tokens in a sequence, crucial for understanding the context within the input text. Two common types of positional encodings are Alibi and RoPE."],
        ["LLMs have demonstrated emergent abilities such as reasoning, planning, decision-making, in-context learning, and answering in zero-shot settings. These abilities are attributed to the large scale of LLMs, even when they are not specifically trained for these attributes."],
        ["The main challenges associated with LLMs include slow training and inference times, extensive hardware requirements, and higher running costs. These challenges have led to research into more efficient architectures, training strategies, and techniques such as parameter-efficient tuning, pruning, quantization, and knowledge distillation."],
        ["Attention mechanisms in LLMs assign weights to input tokens based on their importance, allowing the model to focus on relevant parts of the input. Different types of attention include self-attention, cross-attention, sparse attention, and flash attention. These variations optimize the processing of input sequences, especially in handling long text inputs efficiently."]
    ]
      
    # Ensure the asked question is one of the predefined questions
    selected_ground_truth = None
    for i, eval_question in enumerate(eval_questions):
        if questions == eval_question.lower().strip():
            selected_ground_truth = ground_truths[i]
    
    selected_ground_truth = str(selected_ground_truth)

    # Prepare data for evaluation
    data = {
        "question": [questions],
        "answer": [answers],
        "contexts": [context],
        "ground_truth": [selected_ground_truth]
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

    
    


    