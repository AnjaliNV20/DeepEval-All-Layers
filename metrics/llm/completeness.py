from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_completeness_metric(model=None):
    return GEval(
        name="Completeness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="The answer must cover the important requirements of the user's request.",
        model=model,
        threshold=0.7,
    )
