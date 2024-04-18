# This program tries to leverage langchain to read code from a local folder and then embed it to ChromaDB
# using OpenAI's GPT-4 model.
import argparse
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from splitter import LanguageSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings


# accept two arguments: the path to the folder containing the code and the path of the chromadb database
# otherwise, print the help message
def arg_parser():
    parser = argparse.ArgumentParser(description="Ingest code to ChromaDB", add_help=True)
    parser.add_argument("path", help="Path to the folder containing the code")
    parser.add_argument("chromadb", help="Path to the ChromaDB database")
    return parser.parse_args()


def load_code(path):
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
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", disallowed_special=(), show_progress_bar=True)
    Chroma.from_documents(documents=texts, embedding=embeddings, persist_directory=db_path)


def main():
    args = arg_parser()
    texts = load_code(args.path)
    embed_code(texts, args.chromadb)


if __name__ == "__main__":
    main()
