"""LLM-judge (G-Eval) prompt alignment metric.

Use this when the test case has no separate list of prompt instructions: the
judge reads the user's prompt itself and checks the explicit requirements in
it. The built-in `PromptAlignmentMetric` (prompt_alignment.py) needs an
explicit `prompt_instructions` list instead.
"""

from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_prompt_alignment_geval_metric(model=None):
    return GEval(
        name="Prompt Alignment",
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        criteria="The answer must follow the user's request and its explicit requirements.",
        model=model,
        threshold=0.7,
    )
