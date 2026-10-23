
from deepeval.test_case import LLMTestCase, SingleTurnParams
from deepeval.metrics import GEval

from agents.agentflow import generate_plan
from config.models import get_judge_model
from utils.testcase_loader import get_test_case


def get_plan_quality_metric(model):
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
            SingleTurnParams.ACTUAL_OUTPUT
        ],
        threshold=0.5,
        model=model
    )


def run_plan_quality():

    case = get_test_case("agentic_testing", "TC_SC3_PLAN_QUALITY")

    question = case["prompt"]
    expected_response = case["expected_response"]
    scenario = case["scenario"]

    plan = generate_plan(question)

    evaluation_input = (
        f"User request: {question}\n\n"
        f"Scenario requirement: {scenario}\n\n"
        f"Expected result: {expected_response}"
    )

    test_case = LLMTestCase(
        input=evaluation_input,
        actual_output=plan
    )

    metric = get_plan_quality_metric(get_judge_model())

    metric.measure(test_case)

    print("\n==============================")
    print("Test Case:", "TC_SC3_PLAN_QUALITY")
    print("Prompt:", question)
    print("Scenario:", scenario)
    print("Expected Response:", expected_response)
    print("Generated Plan:", plan)
    print("Metric:", "Plan Quality")
    print("Score:", metric.score)
    print("==============================")

    return metric.score
