from deepeval.metrics import ConversationalGEval
from deepeval.test_case import MultiTurnParams


DEFAULT_CRITERIA = (
    "The assistant should remain relevant to the user's intent "
    "throughout the conversation."
)

# Used by the multi-turn OrangeHRM scenario.
ORANGEHRM_CRITERIA = (
    "The assistant should remain relevant to the user's intent "
    "throughout the multi-turn conversation. It should correctly "
    "understand references to the OrangeHRM login automation, "
    "Page Object, test script, API validation, and GitHub Actions."
)


def get_conversation_relevancy_metric(model=None, criteria=DEFAULT_CRITERIA):
    return ConversationalGEval(
        name="Conversation Relevancy",
        criteria=criteria,
        evaluation_params=[MultiTurnParams.CONTENT],
        model=model,
        threshold=0.7,
    )
