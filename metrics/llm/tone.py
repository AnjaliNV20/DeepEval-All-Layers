from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_tone_metric(model=None):
    return GEval(
        name="Tone",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Evaluate whether the response uses a professional, respectful, and appropriate tone for the user's request.",
        evaluation_steps=[
            "Identify the audience and purpose from the input.",
            "Check whether the response is respectful, professional, and free of blame or dismissive language.",
            "Check whether the tone is appropriate to the situation and remains clear and concise.",
        ],
        model=model,
        threshold=0.7,
    )
