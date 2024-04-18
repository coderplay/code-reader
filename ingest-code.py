import argparse
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from splitter import LanguageSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

# This script is used to ingest code into a ChromaDB database.
# It accepts two arguments: the path to the folder containing the code and the path of the ChromaDB database.


def arg_parser():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Ingest code to ChromaDB", add_help=True)
    parser.add_argument("path", help="Path to the folder containing the code")
    parser.add_argument("chromadb", help="Path to the ChromaDB database")
    return parser.parse_args()


def load_code(path):
    """
    Load code from the given path and split it into chunks.

    Args:
        path (str): Path to the folder containing the code.

    Returns:
        list: List of split chunks.
    """
    loader = GenericLoader.from_filesystem(
        path,
        glob="**/*",
        suffixes=[".go", "*.rs", "*.proto", "*.md"],
        parser=LanguageParser(),
    )
    documents = loader.load()
    splitter = LanguageSplitter()
    return splitter.split_documents(documents)


def embed_code(texts, db_path):
    """
    Embed the given chunks and save them to the ChromaDB database.

    Args:
        texts (list): List of chunks to embed.
        db_path (str): Path to the ChromaDB database.
    """
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", disallowed_special=(), show_progress_bar=True)
    Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=db_path)


def main():
    """
    Main function to parse arguments, load code, and embed code.
    """
    args = arg_parser()
    texts = load_code(args.path)
    embed_code(texts, args.chromadb)


if __name__ == "__main__":
    main()
