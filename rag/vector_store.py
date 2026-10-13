from pathlib import Path

import chromadb

from rag.embeddings import (create_embedding_function)

class VectorStore:

    def __init__(self,db_path: Path):
        self.client = (chromadb.PersistentClient(path=str(db_path)))
        self.embedding_function = (create_embedding_function())
        self.collection = (
            self.client.get_or_create_collection(
                name="playwright_knowledge",
                embedding_function=self.embedding_function
            )
        )

    def add_documents(self,chunks: list[str]):
        if not chunks:
            return
        ids = [
            f"chunk_{index}"
            for index in range(len(chunks))
        ]

        self.collection.upsert(ids=ids,documents=chunks)

    def search(self,query: str,top_k: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        documents = (results.get("documents", [[]])[0])
        distances = (results.get("distances", [[]])[0])
        return [
            {
                "text": document,
                "distance": distance
            }
            for document, distance
            in zip(documents, distances)
        ]