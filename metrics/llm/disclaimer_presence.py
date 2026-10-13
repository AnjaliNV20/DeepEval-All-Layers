from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_disclaimer_presence_metric(model=None):
    return GEval(
        name="Disclaimer Presence",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Evaluate whether the response includes the disclaimer requirement stated in the user's input, allowing semantically equivalent wording.",
        evaluation_steps=[
            "Find the disclaimer requirement stated in the user's input.",
            "Check whether the actual response clearly communicates that warning to the user.",
            "Do not require identical wording for this semantic evaluation; exact required wording is checked separately by the test.",
            "Treat a disclaimer as present when its meaning is explicitly stated in the actual response.",
        ],
        model=model,
        threshold=0.7,
    )
