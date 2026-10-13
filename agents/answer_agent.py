class AnswerAgent:

    def __init__(self, evaluation_model):
        self.evaluation_model = evaluation_model

    def generate(
        self,
        question: str,
        retrieval_context: list[str]
    ) -> str:

        # ==========================================================
        # RAG ANSWER
        # ==========================================================

        if retrieval_context:

            context_text = "\n".join(retrieval_context)

            prompt = f"""
You are a helpful assistant.

Answer the user's question using ONLY
the provided context.

Do not use your own knowledge.

Do not invent information.

If the answer is not available in the
provided context, say:

"The information is not available
in the Playwright knowledge base."

Context:
{context_text}

Question:
{question}

Answer:
"""

            response = self.evaluation_model.generate(prompt)

            if isinstance(response, (list, tuple)):
                return str(response[0])

            return str(response)

        # ==========================================================
        # LLM FALLBACK
        # ==========================================================

        fallback_prompt = f"""
You are a Playwright knowledge-base assistant.

The retrieval system could not find relevant
information for the user's question.

Do NOT use external knowledge.
Do NOT guess.
Do NOT invent an answer.

If the information is not available,
respond exactly:

"The information is not available
in the Playwright knowledge base."

Question:
{question}

Answer:
"""

        response = self.evaluation_model.generate(
            fallback_prompt
        )

        if isinstance(response, (list, tuple)):
            return str(response[0])

        return str(response)