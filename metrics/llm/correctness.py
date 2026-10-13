from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams
 
 
def get_correctness_metric(model):
    return GEval(
        name="Correctness",
        criteria=(
            "Evaluate whether the actual output is factually correct "
            "when compared with the expected output."
        ),
        evaluation_params=[
            SingleTurnParams.ACTUAL_OUTPUT,
            SingleTurnParams.EXPECTED_OUTPUT
        ],
        threshold=0.7,
        model=model
    )