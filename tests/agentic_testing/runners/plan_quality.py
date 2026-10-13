
from deepeval.test_case import LLMTestCase

from utils.testcase_loader import get_test_case
from agents.agentflow import generate_plan
from config.models import get_judge_model
from metrics import get_plan_quality_metric


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
