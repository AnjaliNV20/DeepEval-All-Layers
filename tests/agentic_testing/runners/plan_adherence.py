
from deepeval.test_case import LLMTestCase, ToolCall
from deepeval.tracing import observe, update_current_span
from deepeval.tracing.tracing import current_span_context, trace_manager

from utils.testcase_loader import get_test_case
from agents.agentflow import run_agentflow
from config.models import get_judge_model
from metrics import get_plan_adherence_metric


@observe(type="agent")
def run_traced_agentflow(prompt):
    generated_plan = []

    def capture_generated_plan(message):
        content = message.get("content")
        if (
            not generated_plan
            and isinstance(content, str)
            and content.lstrip().lower().startswith("plan:")
        ):
            generated_plan.append(content.strip())

    answer, tools_used, trace = run_agentflow(
        prompt,
        response_callback=capture_generated_plan
    )

    actual_tools = [
        ToolCall(
            name=step["name"],
            input_parameters=step["input_parameters"],
            output=step["output"]
        )
        for step in trace
    ]

    trace_output = {"final_response": answer}
    if generated_plan:
        trace_output["agent_generated_plan"] = generated_plan[0]

    update_current_span(
        input=prompt,
        output=trace_output,
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


def run_plan_adherence():

    case = get_test_case("agentic_testing", "TC_SC3_PLAN_ADHERENCE")
    question = case["prompt"]
    execution_prompt = (
        f"{question}\n\n"
        "Before calling tools or giving the final answer, begin your first "
        "assistant response with 'Plan:' and provide a concise numbered plan "
        "for this task. Include any required tool calls in that same response. "
        "After receiving tool results, provide the final answer. Do not claim "
        "actions that the available tools cannot perform."
    )

    answer, tools_used, trace, test_case = run_traced_agentflow(execution_prompt)

    metric = get_plan_adherence_metric(get_judge_model())

    metric.measure(test_case)

    print("\n==============================")
    print("Test Case:", "TC_SC3_PLAN_ADHERENCE")
    print("Prompt:", question)
    print("LLM Response:", answer)
    print("Tools Used:", tools_used)
    print("Agent Trace:", trace)
    print("Metric:", "Plan Adherence")
    print("Score:", metric.score)
    print("==============================")

    return metric.score

