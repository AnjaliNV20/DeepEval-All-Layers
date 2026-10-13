import chromadb


def create_embedding_function():

    return (
        chromadb
        .utils
        .embedding_functions
        .DefaultEmbeddingFunction()
    )