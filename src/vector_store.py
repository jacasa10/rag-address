import os

from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

from fetch_context import get_data_from_web, split_documents
from plots import plot_embeddings

os.environ["TOKENIZERS_PARALLELISM"] = "false"

EMBEDDING_MODEL = "thenlper/gte-small"


class VectorDatabase:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            multi_process=True,
            encode_kwargs={"normalize_embeddings": True},
        )

    def create_vector_store_from_text(self, text: str):
        """Create the vector database from text.

        Args:
            text (str): Text to be embedded.

        Returns:
            FAISS: Vector database.
        """
        return FAISS.from_texts(
            text, self.embedding_model, distance_strategy=DistanceStrategy.COSINE
        )

    def create_vector_store_from_documents(self, docs_processed: list[Document]):
        """create vector database from documents.

        Args:
            docs_processed (list[Document]): list of documents

        Returns:
            _type_: Vector database.
        """
        return FAISS.from_documents(
            docs_processed,
            self.embedding_model,
            distance_strategy=DistanceStrategy.COSINE,
        )

    def save_embedding(self, vector_store):
        vector_store.save_local("vector_store")

    def load_embedding(self):
        return FAISS.load_local(
            "vector_store",
            self.embedding_model,
            allow_dangerous_deserialization=True,
        )

    # TODO
    def merge_data_store(self):
        NotImplemented


if __name__ == "__main__":
    thai_context = [
        "https://www.smarty.com/global-address-formatting/thailand-address-format-examples",
        "https://en.wikipedia.org/wiki/Thai_addressing_system",
    ]

    docs = get_data_from_web(thai_context)
    docs_processed = split_documents(512, docs)

    v_db = VectorDatabase()
    vector_store = v_db.load_embedding()
    embedding_model = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        multi_process=True,
        encode_kwargs={"normalize_embeddings": True},
    )

    # vector_store = v_db.create_vector_store_from_documents(docs_processed)

    # v_db.save_embedding(vector_store)
    print(
        plot_embeddings(
            embedding_model, docs_processed, vector_store, user_query="Street 2 Bangkok"
        )
    )
