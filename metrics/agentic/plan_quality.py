from deepeval.metrics import GEval
from deepeval.test_case import SingleTurnParams


def get_plan_quality_metric(model=None):
    return GEval(
        name="Plan Quality",
        criteria=(
            "Evaluate the quality of the plan generated for the requested "
            "Playwright task. The plan should be logical, complete, clear, "
            "and directly relevant to the request. "
            "For this scenario, a good plan should include: "
            "searching the Playwright coding guidelines, "
            "identifying the recommended locator strategy, and "
            "generating a Playwright script based on that guideline. "
            "The plan does not need to use the exact wording of the "
            "expected result."
        ),
        evaluation_params=[
            SingleTurnParams.INPUT,
            SingleTurnParams.ACTUAL_OUTPUT,
        ],
        threshold=0.5,
        model=model,
    )
