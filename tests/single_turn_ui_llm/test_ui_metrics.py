import allure
import ollama
import pytest
from deepeval.test_case import (
    LLMTestCase,
    ConversationalTestCase,
    Turn,
)

from single_turn_ui_llm.metrics.ui_metrics import build_ui_metric_suite
from model_verify.model import get_answer_model_name, get_judge_model
from single_turn_ui_llm.testcases.loader import get_test_case


TEST_CASE_IDS = [
    "TC_SC1_ANSWER_RELEVANCY",
    "TC_SC1_CONVERSATION_RELEVANCY",
]


@pytest.mark.parametrize("testcase_id", TEST_CASE_IDS)
def test_ui_metrics(testcase_id):
    testcase = get_test_case(testcase_id)

    allure.dynamic.title(f"{testcase_id} - UI evaluation")

    prompt = testcase["prompt"]

    actual_output = ollama.chat(
        model=get_answer_model_name(),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    ).message.content

    llm_test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=testcase.get("expected_response"),
        context=[testcase.get("scenario", "")],
        retrieval_context=[
            testcase.get("expected_response") or prompt
        ],
    )

    conversation_test_case = ConversationalTestCase(
        turns=[
            Turn(
                role="user",
                content=prompt,
            ),
            Turn(
                role="assistant",
                content=actual_output,
            ),
        ]
    )

    judge_model = get_judge_model()

    results = {}

    for metric in build_ui_metric_suite(
        testcase,
        model=judge_model,
    ):
        if metric.__class__.__name__ == "ConversationalGEval":
            metric.measure(conversation_test_case)
        else:
            metric.measure(llm_test_case)

        results[metric.__name__] = metric.score

    print(f"\n{testcase_id}")
    print("LLM Response:")
    print(actual_output)

    for name, score in results.items():
        print(f"{name}: {score}")

        allure.attach(
            f"{name}: {score}",
            name=name,
            attachment_type=allure.attachment_type.TEXT,
        )

    allure.attach(
        prompt,
        name="Prompt",
        attachment_type=allure.attachment_type.TEXT,
    )

    allure.attach(
        actual_output,
        name="Actual LLM Response",
        attachment_type=allure.attachment_type.TEXT,
    )

    assert actual_output.strip()
