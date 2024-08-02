import tqdm, nest_asyncio
from langchain_community.document_loaders import WebBaseLoader
from ragas import RunConfig
from ragas.testset.generator import TestsetGenerator
from ragas.evaluation import simple, reasoning, multi_context
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall


RUN_CONFIG = RunConfig()

# generate questions with openai models
generator_llm = ChatOpenAI(model="gpt-3.5-turbo-16k")
critic_llm = ChatOpenAI(model="gpt-4")
embeddings = OpenAIEmbeddings()

generator = TestsetGenerator.from_langchain(generator_llm, critic_llm, embeddings)

# update the resulting question type generator
distributions = {simple:0.5, multi_context:0.4, reasoning:0.1}

testset = generator.generate_with_langchain_docs(pages, 10, distributions, run_config=RUN_CONFIG)

# Allow nested use of asyncio (used by Ragas)
nest_asyncio.apply()

# Disale tqdm locks
tqdm.get_lock().locks = []

QUESTIONS = testset.questions.to_list()
GROUND_TRUTH = testset.ground_truth.to_list()

def perform_evaluation(docs, retriever):
    """
    Perform RAGAS evaluation on test set.
    Args:
        docs: list of documents
    :return:
        Dict[str, float] dictionary of evaluation results
    """
    eval_data = {
        "questions": QUESTIONS,
        "ground_truth": GROUND_TRUTH,
        "context": [],
    }

    # Getting relevant documents for the evaluation dataset
    print(f"Getting contexts for evaluation set")
    for question in tqdm(QUESTIONS):
        eval_data["contexts"].append(
            [doc.page_content for doc in retriever.similarity_search(question, k=3)]
        )
    # RAGAS expects a Dataset object
    dataset = Dataset.from_dict(eval_data)

    print(f"Running evals")
    result = evaluate(
        dataset=dataset,
        metrics=[context_precision, context_recall],
        run_config=RUN_CONFIG,
        raise_exceptions=False,
    )
    return result

