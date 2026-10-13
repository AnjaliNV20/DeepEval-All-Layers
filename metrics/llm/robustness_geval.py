"""LLM-judge (G-Eval) robustness metric.

This is different from `RobustnessMetric` in input_quality.py, which is a
rule-based spelling check. This one asks the judge model whether the answer
copes with typos and vague questions.
"""

from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_robustness_geval_metric(model=None):
    return GEval(
        name="Robustness",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="The answer should handle typos, vague or ambiguous questions appropriately.",
        model=model,
        threshold=0.7,
    )
