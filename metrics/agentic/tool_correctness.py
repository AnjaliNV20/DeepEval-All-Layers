from deepeval.test_case import LLMTestCase, ToolCall, ToolCallParams
from deepeval.metrics import ToolCorrectnessMetric

from agents.agentflow import run_agentflow
from config.models import get_judge_model
from utils.testcase_loader import get_test_case


def get_tool_correctness_metric(model):
    return ToolCorrectnessMetric(
        threshold=0.5,
        evaluation_params=[ToolCallParams.INPUT_PARAMETERS],
        model=model
    )


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


def run_tool_correctness():

    testcase = get_test_case("agentic_testing", "TC_SC3_TOOL_CORRECTNESS")
    prompt = testcase["prompt"]

    answer, tools_used, trace = run_agentflow(prompt)

    actual_tools = build_tool_calls(trace)

    expected_tools = [
        ToolCall(
            name=expected_tool["name"],
            input_parameters=expected_tool.get("input_parameters", {})
        )
        for expected_tool in testcase.get("expected_tools", [])
    ]

    test_case = LLMTestCase(
        input=prompt,
        actual_output=answer,
        tools_called=actual_tools,
        expected_tools=expected_tools
    )

    metric = get_tool_correctness_metric(get_judge_model())

    print(
        "Actual tools_called:",
        [
            {
                "name": tool.name,
                "input_parameters": tool.input_parameters,
                "output": tool.output,
            }
            for tool in actual_tools
        ],
    )
    print(
        "Expected tools:",
        [
            {
                "name": tool.name,
                "input_parameters": tool.input_parameters,
                "output": tool.output,
            }
            for tool in expected_tools
        ],
    )
    print("Metric evaluation_params:", metric.evaluation_params)
    print("Metric threshold:", metric.threshold)
    print("Metric should_exact_match:", metric.should_exact_match)
    print("Metric should_consider_ordering:", metric.should_consider_ordering)

    metric.measure(test_case)

    print("Metric reason:", metric.reason)
    print("Metric verbose breakdown:", metric.verbose_logs)

    print("\n==============================")
    print("Test Case:", "TC_SC3_TOOL_CORRECTNESS")
    print("Prompt:", prompt)
    print("LLM Response:", answer)
    print("Tools Used:", tools_used)
    print("Metric:", "Tool Correctness")
    print("Score:", metric.score)
    print("==============================")

    return metric.score