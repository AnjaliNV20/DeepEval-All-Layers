class ResearchAgent:

    def __init__(self,document_retriever,vector_retriever):
        self.document_retriever = (document_retriever)
        self.vector_retriever = (vector_retriever)

    def search(self,question: str) -> dict:

        # 1. DOCUMENT SEARCH

        document_context = (self.document_retriever.search(question))
        if document_context:
            return {
                "source": "document",
                "retrieval_context": (
                    document_context
                )
            }

        # 2. VECTOR DB SEARCH

        vector_context = (self.vector_retriever.search(question))
        if vector_context:
            return {
                "source": "vector_db",
                "retrieval_context": (
                    vector_context
                )
            }

        # 3. LLM FALLBACK

        return {
            "source": "llm",
            "retrieval_context": []
        }