class SupervisorAgent:

    def __init__(self,research_agent,answer_agent):
        self.research_agent = research_agent
        self.answer_agent = answer_agent

    def run(self,question: str) -> dict:

        research_result = (self.research_agent.search(question))
        source = research_result["source"]
        retrieval_context = (research_result["retrieval_context"])
        answer = self.answer_agent.generate(question,retrieval_context)

        return {
            "source": source,
            "answer": answer,
            "retrieval_context": retrieval_context
        }


def supervisor_agent(
    question,
    research_agent,
    answer_agent
):

    supervisor = SupervisorAgent(
        research_agent,
        answer_agent
    )

    return supervisor.run(question)