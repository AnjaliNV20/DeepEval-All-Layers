
import allure
import ollama
import pytest
from deepeval.test_case import LLMTestCase

from single_turn_ui_llm.metrics.llm_metrics import (
    build_llm_metric_suite,
    build_safety_metric_suite,
    build_custom_metric_suite,
)
from model_verify.model import get_answer_model_name, get_judge_model
from single_turn_ui_llm.testcases.loader import get_test_case


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


def get_llm_response(prompt):
    response = ollama.chat(
        model=get_answer_model_name(),
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    return response.message.content


def create_test_case(testcase, actual_output, is_custom):
    prompt = testcase["prompt"]
    expected = testcase.get("expected_response")

    context = (
        [
            testcase.get("scenario", ""),
            testcase.get("policy_rules", ""),
            testcase.get("required_disclaimer", ""),
            expected or "",
        ]
        if is_custom
        else [
            testcase.get("scenario", ""),
            expected or "",
        ]
    )

    return LLMTestCase(
        input=prompt,
        actual_output=actual_output,
        expected_output=expected,
        context=context,
        retrieval_context=[expected or prompt],
    )


def report_metrics(metrics, test_case, is_custom):
    results = []

    for metric in metrics:
        metric.measure(test_case)

        name = getattr(metric, "name", metric.__class__.__name__)
        score = getattr(metric, "score", None)
        threshold = getattr(metric, "threshold", None)

        success = getattr(metric, "success", None)

        if success is None:
            success = (
                score is not None
                and threshold is not None
                and score >= threshold
            )

        status = "PASS" if success else "FAIL"
        reason = getattr(metric, "reason", None) or (
            "No evaluation reason returned."
        )

        print(
            f"{name}: Score={score}, "
            f"Threshold={threshold}, Status={status}"
        )

        details = (
            f"Score: {score}\n"
            f"Threshold: {threshold}\n"
            f"Status: {status}"
        )

        if is_custom:
            print(f"Reason: {reason}")
            details += f"\nReason: {reason}"

        allure.attach(
            details,
            name=name,
            attachment_type=allure.attachment_type.TEXT,
        )

        results.append((name, score, threshold, status))

    return results


def run_evaluation(testcase_id, metric_builder, evaluation_type):
    testcase = get_test_case(testcase_id)
    prompt = testcase["prompt"]
    is_custom = evaluation_type == "Custom G-Eval"

    allure.dynamic.title(f"{testcase_id} - {evaluation_type}")

    actual_output = get_llm_response(prompt)
    test_case = create_test_case(testcase, actual_output, is_custom)

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

    print(f"\nTest Case: {testcase_id}")
    print("LLM Response:")
    print(actual_output)

    metrics = metric_builder(testcase, model=get_judge_model())
    results = report_metrics(metrics, test_case, is_custom)

    required_disclaimer = testcase.get("required_disclaimer")

    if is_custom and required_disclaimer:
        disclaimer_present = required_disclaimer in actual_output
        status = "PASS" if disclaimer_present else "FAIL"

        print(f"Exact disclaimer check: {status}")

        allure.attach(
            f"Required phrase: {required_disclaimer}\nStatus: {status}",
            name="Disclaimer exact-text check",
            attachment_type=allure.attachment_type.TEXT,
        )

    assert actual_output.strip(), "LLM response is empty"

    if is_custom:
        failures = [
            f"{name}: score={score}, threshold={threshold}"
            for name, score, threshold, status in results
            if status != "PASS"
        ]

        assert not failures, "Custom metric failures: " + "; ".join(failures)

        if required_disclaimer:
            assert required_disclaimer in actual_output, (
                f"Required disclaimer is missing: {required_disclaimer}"
            )


@pytest.mark.parametrize("testcase_id", TEST_CASE_IDS)
def test_llm_metrics(testcase_id):
    run_evaluation(testcase_id, build_llm_metric_suite, "LLM evaluation")


@pytest.mark.parametrize("testcase_id", SAFETY_TEST_CASE_IDS)
def test_safety_metrics(testcase_id):
    run_evaluation(testcase_id, build_safety_metric_suite, "Safety evaluation")


@pytest.mark.parametrize("testcase_id", CUSTOM_METRIC_TEST_CASE_IDS)
def test_custom_metrics(testcase_id):
    run_evaluation(testcase_id, build_custom_metric_suite, "Custom G-Eval")
