import argparse
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# This script is used to query a ChromaDB database. It accepts one argument: the path to the ChromaDB database.
# The script uses the OpenAIEmbeddings and ChatOpenAI models to generate a search query based on the user's input. T
# he search query is then used to retrieve relevant information from the ChromaDB database. The retrieved information
# is used to answer the user's question.


def arg_parser():
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Query code", add_help=True)
    parser.add_argument("chromadb", help="Path to the ChromaDB database")
    return parser.parse_args()


def load_embeddings(chroma_db_path) -> Chroma:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large", disallowed_special=(), show_progress_bar=True)
    return Chroma(persist_directory=chroma_db_path, embedding_function=embeddings)


def main():
    """
    Main function to parse arguments, load ChromaDB, and perform a query.
    """
    args = arg_parser()
    db = load_embeddings(args.chromadb)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8},
    )

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    # First we need a prompt that we can pass into an LLM to generate this search query
    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to "
                "the conversation",
            ),
        ]
    )

    retriever_chain = create_history_aware_retriever(llm, retriever, prompt)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer the user's questions based on the below context:\n\n{context}",
            ),
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
        ]
    )
    document_chain = create_stuff_documents_chain(llm, prompt)

    qa = create_retrieval_chain(retriever_chain, document_chain)
    question = input("Ask a question about the code: ")
    result = qa.invoke({"input": question})
    print(result["answer"])


if __name__ == '__main__':
    main()
