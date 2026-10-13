from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_geval_metric(model=None):
    metric = GEval(
        name="GEval",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="The answer should be relevant, clear, and grounded in the user's request without unsupported claims.",
        model=model,
        threshold=0.7,
    )
    metric.name = "GEval"
    return metric
