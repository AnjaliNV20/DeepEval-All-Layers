import json
from pathlib import Path
 
import allure
import pytest
 
from deepeval import assert_test
from deepeval.test_case import LLMTestCase
 
from config.models import get_evaluation_model
from chatbot.chatbot import generate_actual_output
 
from metrics import (
    get_answer_relevancy_metric,
    get_hallucination_metric,
    get_faithfulness_metric,
    get_correctness_metric,
    get_toxicity_metric,
    get_bias_metric
)
from metrics import get_prompt_alignment_metric
from metrics import SpellingMistakeMetric, RobustnessMetric
 
 
# ---------------------------------------
# Load Test Data
# ---------------------------------------
 
DATA_FILE = (
    Path(__file__).resolve().parents[2]
    / "testdata"
    / "llm_ui_safety"
    / "test_data.json"
)
 
with open(DATA_FILE, "r", encoding="utf-8") as file:
    TEST_DATA = json.load(file)
 
 
# ---------------------------------------
# Evaluation Model
# ---------------------------------------
 
evaluation_model = get_evaluation_model()
 
 
# ---------------------------------------
# Metrics
# ---------------------------------------
 
answer_relevancy = get_answer_relevancy_metric(
    evaluation_model
)
 
hallucination = get_hallucination_metric(
    evaluation_model
)
 
faithfulness = get_faithfulness_metric(
    evaluation_model
)
 
correctness = get_correctness_metric(
    evaluation_model
)
 
toxicity = get_toxicity_metric(
    evaluation_model
)
 
bias = get_bias_metric(
    evaluation_model
)
 
 
# ---------------------------------------
# Test
# ---------------------------------------
 
@pytest.mark.parametrize(
    "data",
    TEST_DATA,
    ids=[data["test_case_id"] for data in TEST_DATA]
)
def test_llm_evaluation(data):
 
    actual_output = generate_actual_output(
        data["input"],
        data["retrieval_context"]
    )
 
    test_case = LLMTestCase(
        input=data["input"],
        actual_output=actual_output,
        expected_output=data["expected_output"],
        context=data["context"],
        retrieval_context=data["retrieval_context"]
    )

    prompt_alignment = get_prompt_alignment_metric(
        evaluation_model,
        data["prompt_instructions"]
    )

    spelling_mistakes = SpellingMistakeMetric(
        reference_texts=data["context"] + [data["expected_output"]]
    )
    spelling_robustness = RobustnessMetric(spelling_mistakes)
 
    metrics = [
        answer_relevancy,
        hallucination,
        faithfulness,
        correctness,
        toxicity,
        bias,
        prompt_alignment,
        spelling_robustness
    ]

    allure.dynamic.title(f"{data['test_case_id']} - {data['input']}")

    allure.attach(data["input"], name="Question", attachment_type=allure.attachment_type.TEXT)
    allure.attach(actual_output, name="Ollama Answer", attachment_type=allure.attachment_type.TEXT)
    allure.attach(
        "\n".join(data["context"]),
        name="Reference Context",
        attachment_type=allure.attachment_type.TEXT
    )
    allure.attach(
        data["expected_output"],
        name="Expected Answer",
        attachment_type=allure.attachment_type.TEXT
    )
    spelling_mistakes.measure(test_case)
    allure.attach(
        (
            f"Score: {spelling_mistakes.score}\n"
            f"Threshold: {spelling_mistakes.threshold}\n"
            f"Passed: {spelling_mistakes.success}\n"
            f"Reason: {spelling_mistakes.reason}"
        ),
        name="Spelling Mistake Report",
        attachment_type=allure.attachment_type.TEXT
    )
    allure.attach(
        "\n".join(data["prompt_instructions"]),
        name="Prompt Instructions",
        attachment_type=allure.attachment_type.TEXT
    )

    try:
        assert_test(
            test_case,
            metrics,
            run_async=False
        )
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
                name=f"{metric.__class__.__name__} Report",
                attachment_type=allure.attachment_type.TEXT
            )