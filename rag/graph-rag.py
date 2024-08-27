from pathlib import Path
from typing import List

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.graphs import Neo4jGraph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai.embeddings import AzureOpenAIEmbeddings
from langchain_openai.chat_models import AzureChatOpenAI
from langchain_text_splitters import TokenTextSplitter
from neo4j.exceptions import ClientError
from nltk.corpus.reader import documents

load_dotenv()
AZURE_OPENAI_API_KEY = process.env[]

txt_path = "C:\\Users\\Hammer\\PycharmProjects\\llm_adons\\data\\dune.txt"

graph = Neo4jGraph()

# Embeddings and LLM model
LLM_MODEL = "gpt35turbo"
EMBEDDING_MODEL = "ada0021_6"

embeddings = AzureOpenAIEmbeddings()
embedding_dimension = 1536
#llm = AzureChatOpenAI(temperature=0)

llm = AzureChatOpenAI(
  openai_api_type="azure",
  openai_api_version="2024-02-01",
  openai_api_key=AZURE_OPENAI_API_KEY,
  azure_endpoint=AZURE_OPENAI_ENDPOINT,
  model=LLM_MODEL,
  temperature=0
)

# Load text file
loader = TextLoader(str(txt_path), encoding='utf-8')
documents = loader.load()

# Ingest Parent-Child node pairs
parent_splitter = TokenTextSplitter(chunk_size=512, chunk_overlap=24)
child_splitter = TokenTextSplitter(chunk_size=100, chunk_overlap=24)
parent_documents = parent_splitter.split_documents(documents)

for i, parent in enumerate(parent_documents):
    child_document = child_splitter.split_documents([parent])
    params = {
        "parent_text": parent.page_content,
        "parent_id": i,
        "parent_embedding": embeddings.embed_query(parent.page_content),
        "children": [
            {
                "text": c.page_content,
                "id": f"{i}-{ic}",
                "embedding": embeddings.embed_query(c.page_content),
            }
            for ic, c in enumerate(child_document)
        ],
    }

    # Ingest data
    graph.query(
        """
    MERGE (p:Parent {id: $parent_id})
    SET p.text = $parent_text
    WITH p
    CALL db.create.setVectorProperty(p, 'embedding', $parent_embedding)
    YIELD node
    WITH p 
    UNWIND $children AS child
    MERGE (c:Child {id: child.id})
    SET c.text = child.text
    MERGE (c)<-[:HAS_CHILD]-(p)
    WITH c, child
    CALL db.create.setVectorProperty(c, 'embedding', child.embedding)
    YIELD node
    RETURN count(*)
    """,
        params,
    )

    # Create vector index for child
    try:
        graph.query(
            "CALL db.index.vector.createNodeIndex('typical_rag', "
            "'Parent', 'embedding', $dimension, 'cosine')",
            {"dimension": embedding_dimension},
        )
    except ClientError as e:
        pass

class Questions(BaseModel):
    """ Generating hypothetical questions based on text"""

    questions: List[str] = Field(
            ...,
            description=(
                "Generated hypothetical questions based on uploaded text"
            ),
    )

questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are generating hypothetical questions based on the information "
                "found in the text. Make sure to provide full context in the generated "
                "questions."
            ),
        ),
        (
            "human",
            (
                "Use the given format to generate hypothtical questions from the "
                f"following input: {input}"
            ),
        ),
    ]
)

question_chain = questions_prompt | llm.with_structured_output(Questions)

for i, parent in enumerate(parent_documents):
    questions = question_chain.invoke(parent.page_content).questions
    params = {
        "parent_id": i,
        "questions": [
            {"text": q, "id": f"{i}-{iq}", "embedding": embeddings.embed_query(q)}
            for iq, q in enumerate(questions)
            if q
        ],
    }
    graph.query(
        """
    MERGE (p:Parent {id: $parent_id})
    WITH p
    UNWIND $questions AS question
    CREATE (q:Question {id: question.id})
    SET q.text = question.text
    MERGE (q)<-[:HAS_QUESTION]-(p)
    WITH q, question
    CALL db.create.setVectorProperty(q, 'embedding', question.embedding)
    YIELD node
    RETURN count(*)
    """,
        params,
    )

    # Create vector index
    try:
        graph.query(
            "CALL db.index.vector.createNodeIndex('hypothetical_questions', "
            "'Question', 'embedding', $dimension, 'cosine')",
            {"dimension": embedding_dimension},
        )
    except ClientError as e:
        pass

# Ingest summaries

summary_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            (
                "You are generating concise and accurate summaries based on the "
                "information found in the text."
            ),
        ),
        (
            "human",
            ("Generate a summary of the following input: {question}\n" "Summary:"),
        ),
    ]
)

summary_chain = summary_prompt | llm

for i, parent in enumerate(parent_documents):
    summary = summary_chain.invoke({"question": parent.page_content}).content
    params = {
        "parent_id": i,
        "summary": summary,
        "embedding": embeddings.embed_query(summary),
    }
    graph.query(
        """
    MERGE (p:Parent {id: $parent_id})
    MERGE (p)-[:HAS_SUMMARY]->(s:Summary)
    SET s.text = $summary
    WITH s
    CALL db.create.setVectorProperty(s, 'embedding', $embedding)
    YIELD node
    RETURN count(*)
    """,
        params,
    )
    # Create vector index
    try:
        graph.query(
            "CALL db.index.vector.createNodeIndex('summary', "
            "'Summary', 'embedding', $dimension, 'cosine')",
            {"dimension": embedding_dimension},
        )
    except ClientError:  # already exists
        pass


