from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_groundedness_metric(model=None):
    return GEval(
        name="Groundedness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.CONTEXT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="The answer must stay within the available facts and must not invent information.",
        model=model,
        threshold=0.7,
    )
