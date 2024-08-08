import os
import chromadb
import nest_asyncio
from datasets import Dataset
from dotenv import load_dotenv
from langchain_community.embeddings import SentenceTransformerEmbeddings
from ragas.testset import TestsetGenerator
from ragas.testset.evolutions import simple, reasoning, multi_context
from ragas import RunConfig
from ragas.metrics import (context_precision, context_recall)
from tqdm import tqdm

from langchain_openai.chat_models import AzureChatOpenAI
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from ragas import evaluate

load_dotenv()

AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
LLM_MODEL = "gpt35turbo"
EMBEDDING_MODEL = "ada0021_6"

storage_path = "../vectordb"

chroma_client = chromadb.PersistentClient(storage_path)
collection = "test"
chroma_vector_search_index_name = "vector_index"

#ds = load_dataset("m-ric/huggingface_doc", split="train")


metrics = [context_precision, context_recall]


llm_1 = AzureChatOpenAI(
    openai_api_type="azure",
    openai_api_version="2024-02-01",
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model=LLM_MODEL,
    temperature=0
    )

llm_2 = AzureChatOpenAI(
    openai_api_type="azure",
    openai_api_version="2024-02-01",
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model=LLM_MODEL,
    temperature=0
)

azure_embeddings = AzureOpenAIEmbeddings(
    openai_api_type="azure",
    openai_api_version="2024-02-01",
    openai_api_key=AZURE_OPENAI_API_KEY,
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    model=EMBEDDING_MODEL,
    allowed_special={'<|endoftext|>'}
)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

def perform_evaluation(docs, retriever):
    """
    Perform RAGAS evaluation on test set.
    Args:
        docs: list of documents
    :return:
        Dict[str, float] dictionary of evaluation results
    """
    #langchain_docs = [LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]}) for doc in tqdm(docs)]
    generator = TestsetGenerator.from_langchain(llm_1, llm_2, azure_embeddings)
    distributions = {simple: 0.5, multi_context: 0.4, reasoning: 0.1}

    testdata = generator.generate_with_langchain_docs(docs, 10, distributions, run_config=RunConfig())
    testdata = testdata.to_pandas()


    # Allow nested use of asyncio (used by Ragas)
    nest_asyncio.apply()

    QUESTIONS = testdata.question.to_list()
    GROUND_TRUTH = testdata.ground_truth.to_list() ## llm answers is compared with ground_truth


    eval_data = {
        "question": QUESTIONS,
        "ground_truth": GROUND_TRUTH,
        "contexts": [],
    }

    # Getting relevant documents for the evaluation dataset
    print(f"Getting contexts for evaluation set")
    for question in tqdm(QUESTIONS):
        eval_data["contexts"].append(
            [doc.page_content for doc in retriever.vectorstore.similarity_search(question, k=1)]
        )
    # RAGAS expects a Dataset object
    dataset = Dataset.from_dict(eval_data)

    print(f"Running evals")
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
        llm=llm_1,
        embeddings=azure_embeddings,
        run_config=RunConfig(),
        raise_exceptions=False,
    )
    chroma_client.delete_collection(collection)
    return result