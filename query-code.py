# This script reads the context from chromedb and then uses it to answer a question through OpenAI.
import argparse
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


def arg_parser():
    parser = argparse.ArgumentParser(description="Query code", add_help=True)
    parser.add_argument("chromadb", help="Path to the ChromaDB database")
    return parser.parse_args()


def main():
    args = arg_parser()
    arg_parser()
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small", disallowed_special=(), show_progress_bar=True)
    db = Chroma(persist_directory=args.chromadb, embedding_function=embeddings)
    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 8},
    )

    llm = ChatOpenAI(model="gpt-4")

    # First we need a prompt that we can pass into an LLM to generate this search query
    prompt = ChatPromptTemplate.from_messages(
        [
            ("placeholder", "{chat_history}"),
            ("user", "{input}"),
            (
                "user",
                "Given the above conversation, generate a search query to look up to get information relevant to the conversation",
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
