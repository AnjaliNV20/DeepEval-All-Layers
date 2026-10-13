from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_policy_compliance_metric(model=None):
    return GEval(
        name="Policy Compliance",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="Evaluate whether the response follows all explicit policy requirements and prohibitions stated in the user's input.",
        evaluation_steps=[
            "Identify the policy requirements and prohibitions stated in the user's input.",
            "Check the actual response for compliance with each identified requirement.",
            "Treat a requirement as satisfied when the response visibly follows it; do not claim a requirement is missing if it is present in the actual response.",
            "Penalize each explicit policy violation and score compliant responses highly.",
        ],
        model=model,
        threshold=0.7,
    )
