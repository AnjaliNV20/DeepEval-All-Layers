from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import observe, update_current_span
from deepeval.tracing.tracing import current_span_context, trace_manager

from utils.testcase_loader import get_test_case
from agents.agentflow import run_agentflow
from config.models import get_judge_model
from metrics import get_task_completion_metric


@observe(type="agent")
def run_traced_agentflow(prompt):
    answer, tools_used, trace = run_agentflow(prompt)

    actual_tools = [
        ToolCall(
            name=step["name"],
            input_parameters=step["input_parameters"],
            output=step["output"]
        )
        for step in trace
    ]

    update_current_span(
        input=prompt,
        output=answer,
        tools_called=actual_tools
    )

    test_case = LLMTestCase(
        input=prompt,
        actual_output=answer,
        tools_called=actual_tools
    )
    current_span = current_span_context.get()
    test_case._trace_dict = trace_manager.create_nested_spans_dict(current_span)

    return answer, tools_used, trace, test_case


def run_task_completion():

    testcase = get_test_case("agentic_testing", "TC_SC3_TOOL_CORRECTNESS")
    prompt = testcase["prompt"]

    answer, tools_used, trace, test_case = run_traced_agentflow(prompt)

    metric = get_task_completion_metric(get_judge_model())

    metric.measure(test_case)

    print("\n==============================")
    print("Test Case:", "TC_SC3_TOOL_CORRECTNESS")
    print("Prompt:", prompt)
    print("LLM Response:", answer)
    print("Tools Used:", tools_used)
    print("Metric:", "Task Completion")
    print("Score:", metric.score)
    print("==============================")

    return metric.score