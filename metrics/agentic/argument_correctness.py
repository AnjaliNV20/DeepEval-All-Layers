from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.metrics import ArgumentCorrectnessMetric

from agentic_testing.testcases.loader import get_test_case
from agentic_testing.tools.agentflow import run_agentflow
from model_verify.model import get_judge_model


def build_tool_calls(trace):
    tool_calls = []

    for step in trace:
        tool_calls.append(
            ToolCall(
                name=step["name"],
                input_parameters=step["input_parameters"],
                output=step["output"]
            )
        )

    return tool_calls


def run_argument_correctness():

    testcase = get_test_case("TC_SC3_TOOL_CORRECTNESS")
    prompt = testcase["prompt"]

    answer, tools_used, trace = run_agentflow(prompt)

    test_case = LLMTestCase(
        input=prompt,
        actual_output=answer,
        tools_called=build_tool_calls(trace)
    )

    metric = ArgumentCorrectnessMetric(
        threshold=0.5,
        model=get_judge_model()
    )

    metric.measure(test_case)

    print("\n==============================")
    print("Test Case:", "TC_SC3_TOOL_CORRECTNESS")
    print("Prompt:", prompt)
    print("LLM Response:", answer)
    print("Tools Used:", tools_used)
    print("Metric:", "Argument Correctness")
    print("Score:", metric.score)
    print("==============================")

    return metric.score