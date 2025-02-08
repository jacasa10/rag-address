from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.utils import DistanceStrategy
from langchain_huggingface import HuggingFaceEmbeddings

EMBEDDING_MODEL = "thenlper/gte-small"


class VectorDatabase:
    def __init__(self):
        self.embedding_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            multi_process=True,
            encode_kwargs={"normalize_embeddings": True},
        )

    def create_vector_store(self, text: str):
        """Create the vector database from context.

        Args:
            text (str): Text to be embedded.

        Returns:
            FAISS: Vector database.
        """
        return FAISS.from_texts(
            text, self.embedding_model, distance_strategy=DistanceStrategy.COSINE
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
    pass
    # text = get_data_from_web("https://en.wikipedia.org/wiki/Thai_addressing_system")
    # chunks_huggingface = chunk_data_huggingface(text)
    # vector_database = VectorDatabase()
    # vector_store = vector_database.create_vector_store(chunks_huggingface)
    # new_vector_db = vector_database.load_embedding()

    # print(new_vector_db)
