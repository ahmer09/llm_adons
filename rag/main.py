from langchain_community.document_loaders import WebBaseLoader

from rag.chunking import fixed_token_split
from rag.ingestion import create_vector_store
from rag.ragas_evaluate import perform_evaluation

if __name__ == '__main__':

    web_loader = WebBaseLoader(
        [
            "https://peps.python.org/pep-0483/",
            "https://peps.python.org/pep-0008/",
            "https://peps.python.org/pep-0257/",
        ]
    )
    docs = web_loader.load()

    for chunk_size in [100, 200, 500, 1000]:
        chunk_overlap = int(0.15*chunk_size)
        print(f"CHUNK SIZE: {chunk_size}")
        print("------ Fixed token with overlap ------")

        splits = fixed_token_split(docs, chunk_size, chunk_overlap)
        vector_store = create_vector_store(splits)
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        result = perform_evaluation(docs, retriever)
        print(
            f"Result: {result}"
        )