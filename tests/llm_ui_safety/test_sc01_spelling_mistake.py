import json
from pathlib import Path

import allure

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from chatbot.chatbot import generate_actual_output
from config.models import get_evaluation_model
from metrics import get_answer_relevancy_metric
from metrics import SpellingMistakeMetric


DATA_FILE = Path(__file__).resolve().parents[2] / "testdata" / "llm_ui_safety" / "test_data.json"

with open(DATA_FILE, "r", encoding="utf-8") as file:
    test_data = next(
        data for data in json.load(file) if data["test_case_id"] == "TC003"
    )


def test_tc003_spelling_mistake_answer_relevancy():
    actual_output = generate_actual_output(
        test_data["input"],
        test_data["retrieval_context"]
    )

    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=actual_output,
        expected_output=test_data["expected_output"],
        context=test_data["context"],
        retrieval_context=test_data["retrieval_context"]
    )

    spelling_metric = SpellingMistakeMetric(
        reference_texts=test_data["context"] + [test_data["expected_output"]]
    )
    answer_relevancy_metric = get_answer_relevancy_metric(
        get_evaluation_model()
    )

    allure.dynamic.title(
        f"{test_data['test_case_id']} - Spelling mistake answer relevancy"
    )
    allure.attach(
        test_data["input"],
        name="Misspelled Prompt",
        attachment_type=allure.attachment_type.TEXT
    )
    allure.attach(
        actual_output,
        name="Chatbot Answer",
        attachment_type=allure.attachment_type.TEXT
    )
    allure.attach(
        test_data["expected_output"],
        name="Expected Answer",
        attachment_type=allure.attachment_type.TEXT
    )

    try:
        assert_test(
            test_case,
            [spelling_metric, answer_relevancy_metric],
            run_async=False
        )
    finally:
        spelling_metric.measure(test_case)
        spelling_report = (
            f"Score: {spelling_metric.score}\n"
            f"Threshold: {spelling_metric.threshold}\n"
            f"Passed: {spelling_metric.success}\n"
            f"Reason: {spelling_metric.reason}"
        )
        allure.attach(
            spelling_report,
            name="Spelling Mistake Metric",
            attachment_type=allure.attachment_type.TEXT
        )

        relevancy_report = (
            f"Score: {getattr(answer_relevancy_metric, 'score', None)}\n"
            f"Threshold: {getattr(answer_relevancy_metric, 'threshold', None)}\n"
            f"Passed: {getattr(answer_relevancy_metric, 'success', None)}\n"
            f"Reason: {getattr(answer_relevancy_metric, 'reason', None)}\n"
            f"Error: {getattr(answer_relevancy_metric, 'error', None)}"
        )
        allure.attach(
            relevancy_report,
            name="Answer Relevancy Metric",
            attachment_type=allure.attachment_type.TEXT
        )
