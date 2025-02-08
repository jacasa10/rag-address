import requests
from bs4 import BeautifulSoup
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer


def get_data_from_web(url: str) -> str:
    """Function to get data from the url and return the text content

    Args:
        url (str): url to get data from

    Returns:
        str: data from the url
    """
    response = requests.get(url)
    soup = BeautifulSoup(response.content, "html.parser")
    content = soup.find(id="bodyContent")
    return content.text


MARKDOWN_SEPARATORS = [
    "\n#{1,6} ",
    "```\n",
    "\n\\*\\*\\*+\n",
    "\n---+\n",
    "\n___+\n",
    "\n\n",
    "\n",
    " ",
    "",
]

EMBEDDING_MODEL_NAME = "thenlper/gte-small"


def chunk_data_length(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512, chunk_overlap=50, length_function=len
    )
    return text_splitter.split_text(text)


def chunk_data_huggingface(text):
    text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME),
        chunk_size=512,
        chunk_overlap=int(512 / 10),
        add_start_index=True,
        strip_whitespace=True,
        separators=MARKDOWN_SEPARATORS,
    )
    return text_splitter.split_text(text)


if __name__ == "__main__":
    text = get_data_from_web("https://en.wikipedia.org/wiki/Thai_addressing_system")
    chunks = chunk_data_length(text)
    chunks_huggingface = chunk_data_huggingface(text)
    print(type(chunks_huggingface))
    for chunk in chunks_huggingface:
        print(chunk)
