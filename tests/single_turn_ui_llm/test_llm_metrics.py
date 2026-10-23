import allure
import ollama
import pytest
 
from deepeval.test_case import LLMTestCase
 
from metrics.single_turn_ui_llm.all_llm_metrix import (
    build_llm_metric_suite,
    build_safety_metric_suite,
    build_custom_metric_suite,
)
 
from config.models import (
    get_answer_model_name,
    get_judge_model,
)
 
from utils.testcase_loader import get_test_case
 
 
TEST_CASE_IDS = [
    "TC_SC1_CORRECTNESS",
    "TC_SC1_COMPLETENESS",
    "TC_SC1_CLARITY",
    "TC_SC1_FAITHFULNESS",
    "TC_SC1_GROUNDEDNESS",
    "TC_SC1_HALLUCINATION",
    "TC_SC1_TOXICITY",
    "TC_SC1_BIAS",
    "TC_SC1_GEVAL",
    "TC_SC1_ROBUSTNESS",
    "TC_SC1_PROMPT_ALIGNMENT",
    "TC_SC1_PROMPT_INJECTION_RESISTANCE",
    "TC_SC1_JAILBREAK_RESISTANCE",
]
 
SAFETY_TEST_CASE_IDS = [
    f"TC_SC1_SAFETY_{i:02d}" for i in range(1, 21)
]
 
CUSTOM_METRIC_TEST_CASE_IDS = [
    "TC_SC1_CUSTOM_TONE",
    "TC_SC1_CUSTOM_POLICY_COMPLIANCE",
    "TC_SC1_CUSTOM_DISCLAIMER_PRESENCE",
]
 
 
@pytest.mark.parametrize(
    "testcase_id",
    TEST_CASE_IDS + SAFETY_TEST_CASE_IDS + CUSTOM_METRIC_TEST_CASE_IDS
)
def test_llm_evaluation(testcase_id):
 
    # 1. Load testcase
    testcase = get_test_case("single_turn_ui_llm", testcase_id)
 
    # 2. Get prompt
    prompt = testcase["prompt"]
 
    # 3. Generate AI response
    response = ollama.chat(
        model=get_answer_model_name(),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
 
    actual_output = response.message.content
 
    # 4. Create DeepEval test case
    expected = testcase.get("expected_response")
 
    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=expected,
        context=[
            testcase.get("scenario", ""),
            testcase.get("policy_rules", ""),
            testcase.get("required_disclaimer", ""),
            expected or "",
        ],
        retrieval_context=[expected or prompt],
    )
 
    # 5. Select metric suite
    if testcase_id in SAFETY_TEST_CASE_IDS:
        metrics = build_safety_metric_suite(
            testcase=testcase,
            model=get_judge_model(),
        )
 
    elif testcase_id in CUSTOM_METRIC_TEST_CASE_IDS:
        metrics = build_custom_metric_suite(
            testcase=testcase,
            model=get_judge_model(),
        )
 
    else:
        metrics = build_llm_metric_suite(
            testcase=testcase,
            model=get_judge_model(),
        )
 
    # 6. Display response
    print("\nTest Case:", testcase_id)
    print("LLM Response:", actual_output)
 
    allure.dynamic.title(testcase_id)
    allure.attach(prompt, name="Prompt",
                  attachment_type=allure.attachment_type.TEXT)
    allure.attach(actual_output, name="LLM Response",
                  attachment_type=allure.attachment_type.TEXT)
 
    # 7. Execute metrics
    for metric in metrics:
 
        metric.measure(test_case)
 
        score = metric.score
        threshold = metric.threshold
 
        status = "PASS" if metric.success else "FAIL"
 
        print("\nMetric:", metric.__class__.__name__)
        print("Score:", score)
        print("Threshold:", threshold)
        print("Status:", status)
 
        allure.attach(
            f"Score: {score}\nThreshold: {threshold}\nStatus: {status}",
            name=metric.__class__.__name__,
            attachment_type=allure.attachment_type.TEXT,
        )
 
    # 8. Validate response
    assert actual_output.strip(), "LLM response is empty"