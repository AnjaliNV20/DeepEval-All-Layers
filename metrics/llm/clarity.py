from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_clarity_metric(model=None):
    return GEval(
        name="Clarity",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="The answer must be clear, organized, and easy to understand.",
        model=model,
        threshold=0.7,
    )
