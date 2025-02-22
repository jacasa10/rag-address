from typing import List, Optional, Tuple

from langchain.docstore.document import Document as LangchainDocument
from langchain_community.vectorstores import FAISS
from omegaconf import OmegaConf
from ragatouille import RAGPretrainedModel
from transformers import Pipeline

from fetch_context import get_data_from_web, split_documents
from llm import RAG_PROMPT_TEMPLATE, READER_LLM, RERANKER
from vector_store import VectorDatabase


def answer_with_rag(
    question: str,
    llm: Pipeline,
    knowledge_index: FAISS,
    reranker: Optional[RAGPretrainedModel] = None,
    num_retrieved_docs: int = 30,
    num_docs_final: int = 5,
) -> Tuple[str, List[LangchainDocument]]:
    # Gather documents with retriever
    print("=> Retrieving documents...")
    relevant_docs = knowledge_index.similarity_search(query=question, k=num_retrieved_docs)
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text

    # Optionally rerank results
    if reranker:
        print("=> Reranking documents...")
        relevant_docs = reranker.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]

    relevant_docs = relevant_docs[:num_docs_final]

    # Build the final prompt
    context = "\nExtracted documents:\n"
    context += "".join([f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)])

    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)

    # Redact an answer
    print("=> Generating answer...")
    answer = llm(final_prompt)[0]["generated_text"]

    return answer, relevant_docs


if __name__ == "main":
    conf = OmegaConf.load("config.yml")
    docs = get_data_from_web(conf["urls"])
    documents = split_documents(512, docs)
    vector_database = VectorDatabase()
    knowledge_vector_database = vector_database.create_vector_store_from_documents(documents)

    question = "parse this address 1234 Main St, Springfield, IL 62701"
    answer, relevant_docs = answer_with_rag(
        question, READER_LLM, knowledge_vector_database, reranker=RERANKER
    )
