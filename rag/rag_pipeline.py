class RAGPipeline:

    def __init__(self,document_retriever,vector_retriever,llm):

        self.document_retriever = (document_retriever)
        self.vector_retriever = (vector_retriever)
        self.llm = llm


    def _normalize_response(self,response) -> str:
        if isinstance(response,(list, tuple)):
            return str(response[0])
        return str(response)


    def generate_answer(self,question: str,context: list[str]) -> str:

        context_text = "\n".join(context)
        prompt = f"""
You are a helpful assistant.

Answer the user's question using only
the provided context.

Do not invent information.

If the context does not contain enough
information, say that the information is
not available in the context.

Context:
{context_text}

Question:
{question}

Answer:
"""
        response = self.llm.generate(prompt)
        return self._normalize_response(response)

    def run(self,question: str) -> dict:

        # 1. EXTERNAL DOCUMENT
   

        document_context = (self.document_retriever.search(question))

        if document_context:

            answer = self.generate_answer(question,document_context)

            return {"source": "document","answer": answer,"retrieval_context": document_context
            }

        # 2. VECTOR DB

        vector_context = (self.vector_retriever.search(question))
        if vector_context:
            answer = self.generate_answer(question,vector_context)

            return {"source": "vector_db","answer": answer,"retrieval_context": vector_context
            }

        # 3. LLM FALLBACK
    
        response = self.llm.generate(question)
        answer = self._normalize_response(response)

        return {"source": "llm","answer": answer,"retrieval_context": []
        }