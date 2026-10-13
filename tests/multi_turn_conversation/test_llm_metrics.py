import ollama

from deepeval.test_case import LLMTestCase

from config.models import (
    get_answer_model_name,
    get_judge_model,
)

from metrics import build_llm_metric_suite
from utils.testcase_loader import get_test_case


TEST_CASE_IDS = [
    "TC_SC2_P1",
    "TC_SC2_P2",
    "TC_SC2_P3",
    "TC_SC2_P4",
    "TC_SC2_P5",
]


def test_multi_turn_conversation():

    # 1. Conversation history
    messages = []

    # 2. Run P1 → P5 sequentially
    for testcase_id in TEST_CASE_IDS:

        # 3. Load current testcase from JSON
        testcase = get_test_case("multi_turn_conversation", testcase_id)

        # 4. Get current prompt
        prompt = testcase["prompt"]

        # 5. Add current user prompt to SAME conversation
        messages.append({
            "role": "user",
            "content": prompt
        })

        # 6. Send COMPLETE conversation to Answer LLM
        response = ollama.chat(
            model=get_answer_model_name(),
            messages=messages
        )

        # 7. Get AI response
        actual_output = response.message.content

        # 8. Add AI response to SAME conversation
        messages.append({
            "role": "assistant",
            "content": actual_output
        })

        # 9. Create DeepEval test case
        test_case = LLMTestCase(
            input=prompt,
            actual_output=actual_output,
            expected_output=testcase.get("expected_response"),
            context=[
                testcase.get("scenario", "")
            ],
            retrieval_context=[
                testcase.get("scenario", "")
            ]
        )

        # 10. Get judge model
        judge_model = get_judge_model()

        # 11. Build metrics
        metrics = build_llm_metric_suite(
            testcase=testcase,
            model=judge_model
        )

        # 12. Evaluate metrics
        for metric in metrics:

            metric.measure(test_case)

            print("\n-----------------------------")
            print("Test Case:", testcase_id)
            print("Prompt:", prompt)
            print("LLM Response:", actual_output)
            print("Metric:", metric.__class__.__name__)
            print("Score:", metric.score)
            print("-----------------------------")

        # 13. Basic response validation
        assert actual_output.strip()