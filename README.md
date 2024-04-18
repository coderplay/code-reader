# Documentation

## Pre-requisites
* Setup the OPENAI_API_KEY in your environment variables

## `ingest-code.py`

This script is used to ingest code into a ChromaDB database. It accepts two arguments: the path to the folder containing the code and the path of the ChromaDB database.

The script performs the following steps:

1. Parses command line arguments to get the path to the code and the ChromaDB database.
2. Loads the code from the given path and splits it into chunks using the `LanguageSplitter` class.
3. Embeds the chunks using the `OpenAIEmbeddings` model and saves them to the ChromaDB database.



## `query-code.py`

This script is used to query a ChromaDB database. It accepts one argument: the path to the ChromaDB database. The script uses the `OpenAIEmbeddings` and `ChatOpenAI` models to generate a search query based on the user's input. The search query is then used to retrieve relevant information from the ChromaDB database. The retrieved information is used to answer the user's question.

The script performs the following steps:

1. Parses command line arguments to get the path to the ChromaDB database.
2. Loads the embeddings from the ChromaDB database.
3. Creates a retriever to retrieve information from the database.
4. Creates a prompt to generate a search query based on the user's input.
5. Creates a retrieval chain to retrieve information based on the search query.
6. Asks the user a question about the code.
7. Invokes the retrieval chain with the user's question and prints the answer.

