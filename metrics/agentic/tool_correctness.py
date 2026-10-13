from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import ToolCallParams


def get_tool_correctness_metric(model=None):
    return ToolCorrectnessMetric(
        threshold=0.5,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        model=model,
    )
