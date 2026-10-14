import json
from pathlib import Path

import allure

from deepeval import assert_test
from deepeval.test_case import LLMTestCase

from chatbot.chatbot import generate_ui_output
from config.models import get_evaluation_model
from metrics import get_harmfulness_metric
from metrics import get_violence_detection_metric


DATA_FILE = Path(__file__).resolve().parents[2] / "testdata" / "llm_ui_safety" / "test_data.json"

with open(DATA_FILE, "r", encoding="utf-8") as file:
    test_data = next(
        data for data in json.load(file) if data["test_case_id"] == "TC_006"
    )


def test_tc006_refuses_bomb_creation_request():
    actual_output = generate_ui_output(
        test_data["input"],
        test_data["prompt_instructions"]
    )

    test_case = LLMTestCase(
        input=test_data["input"],
        actual_output=actual_output,
        expected_output=test_data["expected_output"],
        context=test_data["context"]
    )

    evaluation_model = get_evaluation_model()
    metrics = [
        get_harmfulness_metric(evaluation_model),
        get_violence_detection_metric(evaluation_model)
    ]

    allure.dynamic.title(
        f"{test_data['test_case_id']} - Refuse bomb creation request"
    )
    allure.attach(
        test_data["input"],
        name="User Prompt",
        attachment_type=allure.attachment_type.TEXT
    )
    allure.attach(
        actual_output or "<empty response>",
        name="Chatbot Response",
        attachment_type=allure.attachment_type.TEXT
    )
    allure.attach(
        test_data["expected_output"],
        name="Expected Safe Response",
        attachment_type=allure.attachment_type.TEXT
    )

    try:
        assert_test(test_case, metrics, run_async=False)
    finally:
        for metric in metrics:
            report = (
                f"Score: {getattr(metric, 'score', None)}\n"
                f"Threshold: {getattr(metric, 'threshold', None)}\n"
                f"Passed: {getattr(metric, 'success', None)}\n"
                f"Reason: {getattr(metric, 'reason', None)}\n"
                f"Error: {getattr(metric, 'error', None)}"
            )
            allure.attach(
                report,
                name=f"{metric.__name__} Report",
                attachment_type=allure.attachment_type.TEXT
            )
