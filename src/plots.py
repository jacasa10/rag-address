import numpy as np
import pacmap
import pandas as pd
import plotly.express as px
from langchain.docstore.document import Document
from langchain_community.embeddings import HuggingFaceEmbeddings


def plot_embeddings(
    embedding_model: HuggingFaceEmbeddings,
    docs_processed: list[Document],
    KNOWLEDGE_VECTOR_DATABASE,
    user_query: None | str = None,
):
    # Embed a user query in the same space
    if user_query:
        query_vector = embedding_model.embed_query(user_query)

    embedding_projector = pacmap.PaCMAP(
        n_components=2, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, random_state=1
    )

    embeddings_2d = [
        list(KNOWLEDGE_VECTOR_DATABASE.index.reconstruct_n(idx, 1)[0])
        for idx in range(len(docs_processed))
    ]
    if query_vector:
        embeddings_2d = embeddings_2d + [query_vector]

    # Fit the data (the index of transformed data corresponds to the index of the original data)
    documents_projected = embedding_projector.fit_transform(np.array(embeddings_2d), init="pca")

    df = pd.DataFrame.from_dict(
        [
            {
                "x": documents_projected[i, 0],
                "y": documents_projected[i, 1],
                "source": docs_processed[i].metadata["source"].split("/")[1],
                "extract": docs_processed[i].page_content[:100] + "...",
                "symbol": "circle",
                "size_col": 4,
            }
            for i in range(len(docs_processed))
        ]
        + [
            {
                "x": documents_projected[-1, 0],
                "y": documents_projected[-1, 1],
                "source": "User query",
                "extract": user_query,
                "size_col": 100,
                "symbol": "star",
            }
        ]
    )

    # Visualize the embedding
    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="source",
        hover_data="extract",
        size="size_col",
        symbol="symbol",
        color_discrete_map={"User query": "black"},
        width=1000,
        height=700,
    )
    fig.update_traces(
        marker=dict(opacity=1, line=dict(width=0, color="DarkSlateGrey")),
        selector=dict(mode="markers"),
    )
    fig.update_layout(
        legend_title_text="<b>Chunk source</b>",
        title="<b>2D Projection of Chunk Embeddings via PaCMAP</b>",
    )
    fig.show()


if __name__ == "__main__":
    pass
