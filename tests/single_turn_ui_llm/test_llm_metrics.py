
import allure
import ollama
import pytest
from deepeval.test_case import LLMTestCase

from metrics import (
    build_custom_metric_suite,
    build_llm_metric_suite,
    build_safety_metric_suite,
)
from config.models import get_answer_model_name, get_judge_model
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
    "TC_SC1_SAFETY_01",
    "TC_SC1_SAFETY_02",
    "TC_SC1_SAFETY_03",
    "TC_SC1_SAFETY_04",
    "TC_SC1_SAFETY_05",
    "TC_SC1_SAFETY_06",
    "TC_SC1_SAFETY_07",
    "TC_SC1_SAFETY_08",
    "TC_SC1_SAFETY_09",
    "TC_SC1_SAFETY_10",
    "TC_SC1_SAFETY_11",
    "TC_SC1_SAFETY_12",
    "TC_SC1_SAFETY_13",
    "TC_SC1_SAFETY_14",
    "TC_SC1_SAFETY_15",
    "TC_SC1_SAFETY_16",
    "TC_SC1_SAFETY_17",
    "TC_SC1_SAFETY_18",
    "TC_SC1_SAFETY_19",
    "TC_SC1_SAFETY_20",
]

CUSTOM_METRIC_TEST_CASE_IDS = [
    "TC_SC1_CUSTOM_TONE",
    "TC_SC1_CUSTOM_POLICY_COMPLIANCE",
    "TC_SC1_CUSTOM_DISCLAIMER_PRESENCE",
]


@pytest.mark.parametrize("testcase_id", TEST_CASE_IDS)
def test_llm_metrics(testcase_id):
    testcase = get_test_case("single_turn_ui_llm", testcase_id)

    allure.dynamic.title(f"{testcase_id} - LLM evaluation")

    prompt = testcase["prompt"]

    actual_output = ollama.chat(
        model=get_answer_model_name(),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    ).message.content

    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=testcase.get("expected_response"),
        context=[
            testcase.get("scenario", ""),
            testcase.get("expected_response", ""),
        ],
        retrieval_context=[
            testcase.get("expected_response") or prompt
        ],
    )

    judge_model = get_judge_model()

    results = []

    for metric in build_llm_metric_suite(
        testcase,
        model=judge_model,
    ):
        metric.measure(test_case)

        metric_name = getattr(metric, "name", metric.__class__.__name__)
        score = getattr(metric, "score", None)
        threshold = getattr(metric, "threshold", None)
        status = "PASS" if getattr(metric, "success", score is not None and threshold is not None and score >= threshold) else "FAIL"

        results.append((metric_name, score, threshold, status))

    print(f"\n{testcase_id}")
    print("LLM Response:")
    print(actual_output)

    for name, score, threshold, status in results:
        print(f"{name}: Score={score}, Threshold={threshold}, {status}")

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

    for name, score, threshold, status in results:
        allure.attach(
            f"{name}: Score={score}, Threshold={threshold}, {status}",
            name=name,
            attachment_type=allure.attachment_type.TEXT,
        )

    assert actual_output.strip()


@pytest.mark.parametrize("testcase_id", SAFETY_TEST_CASE_IDS)
def test_safety_metrics(testcase_id):
    testcase = get_test_case("single_turn_ui_llm", testcase_id)

    allure.dynamic.title(f"{testcase_id} - Safety evaluation")

    prompt = testcase["prompt"]

    actual_output = ollama.chat(
        model=get_answer_model_name(),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    ).message.content

    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=testcase.get("expected_response"),
        context=[
            testcase.get("scenario", ""),
            testcase.get("expected_response", ""),
        ],
        retrieval_context=[
            testcase.get("expected_response") or prompt
        ],
    )

    judge_model = get_judge_model()

    results = []

    for metric in build_safety_metric_suite(
        testcase,
        model=judge_model,
    ):
        metric.measure(test_case)

        metric_name = getattr(metric, "name", metric.__class__.__name__)
        score = getattr(metric, "score", None)
        threshold = getattr(metric, "threshold", None)
        status = "PASS" if getattr(metric, "success", score is not None and threshold is not None and score >= threshold) else "FAIL"

        results.append((metric_name, score, threshold, status))

    print(f"\n{testcase_id}")
    print("LLM Response:")
    print(actual_output)

    for name, score, threshold, status in results:
        print(f"{name}: Score={score}, Threshold={threshold}, {status}")

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

    for name, score, threshold, status in results:
        allure.attach(
            f"{name}: Score={score}, Threshold={threshold}, {status}",
            name=name,
            attachment_type=allure.attachment_type.TEXT,
        )

    assert actual_output.strip()


@pytest.mark.parametrize("testcase_id", CUSTOM_METRIC_TEST_CASE_IDS)
def test_custom_metrics(testcase_id):
    testcase = get_test_case("single_turn_ui_llm", testcase_id)

    allure.dynamic.title(f"{testcase_id} - Custom G-Eval")

    prompt = testcase["prompt"]
    actual_output = ollama.chat(
        model=get_answer_model_name(),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    ).message.content

    expected_response = testcase.get("expected_response")
    test_case = LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=expected_response,
        context=[
            testcase.get("scenario", ""),
            testcase.get("policy_rules", ""),
            testcase.get("required_disclaimer", ""),
            expected_response or "",
        ],
        retrieval_context=[expected_response or prompt],
    )

    judge_model = get_judge_model()
    results = []

    for metric in build_custom_metric_suite(testcase, model=judge_model):
        metric.measure(test_case)

        metric_name = getattr(metric, "name", metric.__class__.__name__)
        score = getattr(metric, "score", None)
        threshold = getattr(metric, "threshold", None)
        success = getattr(metric, "success", None)
        if success is None:
            success = score is not None and threshold is not None and score >= threshold
        status = "PASS" if success else "FAIL"
        reason = getattr(metric, "reason", None) or "No evaluation reason returned."
        results.append((metric_name, score, threshold, status, reason))

    print(f"\n{testcase_id}")
    print("LLM Response:")
    print(actual_output)

    for name, score, threshold, status, reason in results:
        print(f"{name}: Score={score}, Threshold={threshold}, {status}")
        print(f"Reason: {reason}")

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

    for name, score, threshold, status, reason in results:
        allure.attach(
            f"Score={score}, Threshold={threshold}, Status={status}\nReason: {reason}",
            name=f"{name} result",
            attachment_type=allure.attachment_type.TEXT,
        )

    required_disclaimer = testcase.get("required_disclaimer")
    if required_disclaimer:
        disclaimer_present = required_disclaimer in actual_output
        disclaimer_status = "PASS" if disclaimer_present else "FAIL"
        print(f"Exact disclaimer check: {disclaimer_status}")
        allure.attach(
            f"Required phrase: {required_disclaimer}\nStatus: {disclaimer_status}",
            name="Disclaimer exact-text check",
            attachment_type=allure.attachment_type.TEXT,
        )

    assert actual_output.strip()
    failed_metrics = [
        f"{name}: score={score}, threshold={threshold}"
        for name, score, threshold, status, _ in results
        if status != "PASS"
    ]
    assert not failed_metrics, "Custom metric failures: " + "; ".join(failed_metrics)
    if required_disclaimer:
        assert required_disclaimer in actual_output, (
            f"Required disclaimer is missing: {required_disclaimer}"
        )

